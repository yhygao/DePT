import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.07, student_temp=0.1, center_momentum=0.9, layerwise_weight=False, tau=1., prompt_div=False, args=None):
        super().__init__()
    
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.prompt_div = prompt_div
        self.teacher_init_temp = args.learn.teacher_init_temp
        self.warmup_epoch = args.learn.warmup_epoch

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(self.teacher_init_temp, teacher_temp, self.warmup_epoch),
            np.ones(args.learn.epochs - self.warmup_epoch) * teacher_temp
        ))  

        self.register_buffer("center", torch.zeros(1, out_dim))
    
        if layerwise_weight:
            weight_list = np.array(list(range(args.model.stage_num))) + 1 
            weight_list = np.exp(weight_list / tau)
            weight_list /= weight_list.sum()
            self.weight_list = weight_list
        else:
            self.weight_list = np.ones(args.model.stage_num) / args.model.stage_num

        self.args = args

    def two_consistency(self, student_out, teacher_out, consistency='ce'):
        total_loss = 0 
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: 
                    continue
    
                if consistency == 'ce':
                    #loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    loss = self.ce_consistency_loss(student_out[v], q)
                elif consistency == 'cos':
                    loss  = self.cos_consistency_loss(student_out[v], q)
                total_loss += loss.mean()
        return total_loss

    def single_consistency(self, student_out, teacher_out, consistency='ce'):
        if consistency == 'ce':
            loss = self.ce_consistency_loss(student_out, teacher_out)
        elif consistency == 'cos':
            loss = self.cos_consistency_loss(student_out, teacher_out)

        return loss.mean()

    def ce_consistency_loss(self, student_out, teacher_out):
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        return loss.mean()

    def cos_consistency_loss(self, student_out, teacher_out):
        return 1 - (F.normalize(student_out, dim=-1, p=2) * F.normalize(teacher_out, dim=-1, p=2)).sum(dim=-1)

    def prompt_diversity_loss(self, prompt, dist_type='cosine'):
        if dist_type == 'euclidean':
            div_loss = torch.cdist(prompt, prompt)
        elif dist_type == 'cosine':
            div_loss = cos_distance(prompt, prompt)
        else:
            raise NotImplementedError(f"{dist_type} distance is not implemented")
        mask = torch.ones(div_loss.shape[-1]) - torch.eye(div_loss.shape[-1])
        mask = mask.to(prompt.device)
        div_loss = div_loss * mask.unsqueeze(0)

        return -div_loss.mean()

    def forward(self, student_output_cls, teacher_output_cls, student_output_prompt=None, teacher_output_prompt=None, epoch=0):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        cls_loss = 0
        prompt_loss = 0
        prompt_div_loss = 0

        if 'cls' in self.args.model.consistency_type:

            student_out_cls = student_output_cls / self.student_temp
            #student_out_cls = student_out_cls.chunk(2)
            #student_out_cls = student_output_cls.chunk(2)

            # teacher centering and sharpening
            ttemp = self.teacher_temp_schedule[epoch]
            teacher_out_cls = F.softmax((teacher_output_cls - self.center) / ttemp, dim=-1)
            #teacher_out_cls = teacher_out_cls.detach().chunk(2)
            #teacher_out_cls = teacher_output_cls.detach().chunk(2)

            #cls_loss = self.two_consistency(student_out_cls, teacher_out_cls, consistency='cos')
            cls_loss = self.single_consistency(student_out_cls, teacher_out_cls, consistency='ce')

            self.update_center(teacher_output_cls.unsqueeze(1), idx=-1)


        if student_output_prompt is not None and teacher_output_prompt is not None:
            if not isinstance(student_output_prompt, list):
                student_output_prompt = [student_output_prompt]
                teacher_output_prompt = [teacher_output_prompt]

            for i in range(len(student_output_prompt)):
                if len(student_output_prompt) == 1: # in case of non-hierarchy
                    weight = 1
                else:
                    weight = self.weight_list[i]

                temp_student_out = student_output_prompt[i]

                if self.prompt_div:
                    prompt_div_loss += self.prompt_diversity_loss(temp_student_out) * weight


                B, PN, dim = temp_student_out.shape
                temp_student_out = temp_student_out.view(B*PN, dim)
                #temp_student_out = temp_student_out.chunk(2)

                temp_teacher_out = teacher_output_prompt[i]
                temp_teacher_out = temp_teacher_out.view(B*PN, dim).detach()
                #temp_teacher_out = temp_teacher_out.chunk(2)

                prompt_loss += self.single_consistency(temp_student_out, temp_teacher_out, consistency='cos') * weight


        return cls_loss, prompt_loss, prompt_div_loss


    @torch.no_grad()
    def update_center(self, teacher_output, idx):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=False)

        if self.args.distributed:
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        if idx == -1:
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        else:
            center = getattr(self, "prompt_center%d"%idx) * self.center_momentum + batch_center * (1 - self.center_momentum)
            setattr(self, "prompt_center%d"%idx, center)


def classification_loss(logits_s, target_labels, args):
    #target_labels = torch.cat([target_labels, target_labels], dim=0)

    loss_cls = cross_entropy_loss(logits_s, target_labels, args)
    accuracy = calculate_acc(logits_s, target_labels)
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def cos_distance(X, Y):
    # X: B, M, D
    # Y: B, N, D
    # Out: B, M, N
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)
    sim = torch.bmm(X, Y.permute(0, 2, 1))

    return 1 - sim

def diversification_loss(logits_w, logits_s, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div

def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100 
    return accuracy
