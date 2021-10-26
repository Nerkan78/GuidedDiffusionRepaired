import unittest
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def create_argparser(args):
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser(args.split())
    add_dict_to_argparser(parser, defaults)
    return parser

class ModelTestCase(unittest.TestCase):
    def test_valid_model(self):
        args1 = '--batch_size 4 --num_samples 100 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True'

        args = create_argparser(args1).parse_args()
        args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        model, diffusion = create_model_and_diffusion(**args_dict)
        self.assertEqual(args_dict['image_size'], model.image_size)
        self.assertEqual(args_dict['num_channels'], model.model_channels)
        self.assertEqual(args_dict['num_heads'], model.num_heads)
        self.assertEqual(args_dict['num_res_blocks'], model.num_res_blocks)
        self.assertEqual(tuple(args_dict['image_size']//int(res) for res in args_dict['attention_resolutions'].split(",")), model.attention_resolutions)

    def test_valid_result(self):

        args1 = '--batch_size 4 --num_samples 100 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True'

        args = create_argparser(args1).parse_args()
        args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        model, diffusion = create_model_and_diffusion(**args_dict)

        model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.eval()


        classifier = create_classifier(**args_to_dict(args1, classifier_defaults().keys()))
        classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.eval()


        def cond_fn(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

        def model_fn(x, t, y=None):
            assert y is not None
            return model(x, t, y if args.class_cond else None)

        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

        sample_fn = diffusion.p_sample_loop
        
        sample = sample_fn(
            model_fn,
            (1, 3, 128, 128),
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=None,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        self.assertEqual(128, sample.shape[1])

    def test_boundary(self):
        im_size = 513
        args1 = '--batch_size 4 --num_samples 100 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True'

        args = create_argparser(args1).parse_args()
        args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        args_dict.pop('image_size', None)
        with self.assertRaises(ValueError):
            create_model_and_diffusion(im_size, **args_dict)
        im_size = 63
        with self.assertRaises(ValueError):
            create_model_and_diffusion(im_size, **args_dict)


        

if __name__ == '__main__':
    unittest.main()
