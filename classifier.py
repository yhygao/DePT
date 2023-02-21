import logging
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import pdb

class ViT_DePT(nn.Module):
    def __init__(self, args, checkpoint_path=None, post_load=False):
        super().__init__()
        self.args = args
        
        #model = models.__dict__[args.arch](pretrained=True)
        model = timm.create_model(args.arch, pretrained=True)

        self.patch_embed = model.patch_embed

        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
       
        self.encoder = model.blocks
        self.norm = model.norm

        self.output_dim = model.head.in_features
        if args.num_classes == 1000:
            self.head = model.head
        else:
            self.head = nn.Linear(self.output_dim, args.num_classes)

        if checkpoint_path and not post_load:
            self.load_from_checkpoint(checkpoint_path)
        
       
        assert args.stage_num in [0, 1, 2, 3, 4, 6, 12]
        
        self.encoder_list = nn.ModuleList()
        
        if args.stage_num == 0:
            self.encoder_list.append(nn.Sequential(self.encoder))

            del self.encoder
            logging.info('Initializing ViT model')
            return
        
        logging.info('Initializing visual prompt model')
        logging.info(f"num prompt: {args.prompt_num}")
        self.prompt_list = nn.ParameterList()
 
        num_layer_each_stage = len(self.encoder) // args.stage_num

        for i in range(args.stage_num):
            if args.init == 'zero':
                param = nn.Parameter(torch.zeros(1, args.prompt_num, self.cls_token.shape[-1]))
            elif args.init == 'randn':
                param = nn.Parameter(torch.randn(1, args.prompt_num, self.cls_token.shape[-1]))
            elif args.init == 'xavier_uniform':
                param = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, args.prompt_num, self.cls_token.shape[-1])))
            elif args.init == 'xavier_normal':
                param = nn.Parameter(nn.init.xavier_normal_(torch.zeros(1, args.prompt_num, self.cls_token.shape[-1])))
            else:
                raise NameError


            self.prompt_list.append(param)
            self.encoder_list.append(
                nn.Sequential(self.encoder[i*num_layer_each_stage: (i+1)*num_layer_each_stage])
            )

        del self.encoder

        if checkpoint_path and post_load:
            self.load_from_checkpoint(checkpoint_path)


    def forward(self, x, return_feats=False):
        
        x = self.patch_embed(x)
        n = x.shape[0]

        batch_class_token = self.cls_token.expand(n, -1, -1)

        x = torch.cat([batch_class_token, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        if self.args.stage_num == 0:
            x = self.encoder_list[0](x)
            x = self.norm(x)
            feat = x[:, 0]
            prompt_updated_list = None
        else:
            batch_prompt = self.prompt_list[0].expand(n, -1, -1)
            x = torch.cat([batch_prompt, x], dim=1)
            x = self.encoder_list[0](x)

            prompt_updated_list = []

            for i in range(1, self.args.stage_num):

                prompt_updated_list.append(x[:, :self.args.prompt_num])
                batch_prompt = self.prompt_list[i].expand(n, -1, -1)
                x = x[:, self.args.prompt_num:]
                x = torch.cat([batch_prompt, x], dim=1)
                x = self.encoder_list[i](x)

            x = self.norm(x)

            prompt_updated_list.append(x[:, :self.args.prompt_num])
            feat = x[:, self.args.prompt_num]


        x = self.head(feat)

        if return_feats:
            return feat, x, prompt_updated_list
        else:
            return x

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=True)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_prompt_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """

        self.encoder_list.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        self.norm.requires_grad_(False)
        self.patch_embed.requires_grad_(False)



        backbone_params = []
        extra_params = []
        if self.args.stage_num > 0:
            backbone_params.extend(self.prompt_list)
        extra_params.extend(self.head.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params
    
    def get_full_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        
        backbone_params.extend(self.encoder_list.parameters())
        backbone_params.append(self.cls_token)
        backbone_params.append(self.pos_embed)
        backbone_params.extend(self.norm.parameters())
        backbone_params.extend(self.patch_embed.parameters())
        
        if self.args.stage_num > 0:
            extra_params.extend(self.prompt_list)
        extra_params.extend(self.head.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params
    
    def get_shot_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        
        backbone_params.extend(self.encoder_list.parameters())
        backbone_params.append(self.cls_token)
        backbone_params.append(self.pos_embed)
        backbone_params.extend(self.norm.parameters())
        backbone_params.extend(self.patch_embed.parameters())
        
        if self.args.stage_num > 0:
            extra_params.extend(self.prompt_list)

        self.head.requires_grad_(False)

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params


    def get_ln_params(self):

        self.encoder_list.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        self.patch_embed.requires_grad_(False)
        self.head.requires_grad_(False)
        
        backbone_params = []
        extra_params = []

        self.norm.requires_grad_(True)
        backbone_params.extend(self.norm.parameters())
        for module in list(self.encoder_list.modules()):
            if isinstance(module, nn.LayerNorm):
                print('has ln')
                module.requires_grad_(True)
                backbone_params.extend(module.parameters())

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.head.weight.shape[0]

