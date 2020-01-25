from collections import deque

import cv2
import numpy
import torch
from cv2 import VideoWriter, VideoWriter_fourcc

import agnes

cv2.ocl.setUseOpenCL(False)


class VisualizeAttention:
    def __init__(self, env, runner, prerun=0, seconds=20, layer_num=0):
        self.nnet = runner.trainer.get_nn_instance()

        self.env = env['env']
        self.state = self.env.reset()
        self.hidden = None
        with torch.no_grad():
            for i in range(prerun):
                t_state = torch.cuda.FloatTensor(self.state)
                dist, self.hidden, value = self.nnet.forward(t_state, self.hidden)
                dist = self.nnet.wrap_dist(dist)
                action = dist.sample().detach().cpu().numpy()
                self.state, done, _, _ = self.env.step(action)
                if done.item():
                    self.hidden = None

        self.obs = self.env.observation_space.shape
        self.width = self.obs[0]*5
        self.height = self.obs[1]*5
        self.FPS = 20
        self.seconds = seconds

        self.outputs = deque(maxlen=10)
        self.gradients = deque(maxlen=10)

        target_layer = self._iterate(self.nnet)[layer_num]

        target_layer.register_forward_hook(self._save_output)
        target_layer.register_backward_hook(self._save_gradient)

    def _iterate(self, nnet):
        lst = []
        for param in nnet.children():
            if isinstance(param, torch.nn.Conv2d):
                lst.append(param)
            else:
                if isinstance(param, torch.nn.Sequential) or isinstance(param, torch.nn.Module):
                    lst.extend(self._iterate(param))
        return lst

    def run(self, filename='AttentionMap'):
        fourcc = VideoWriter_fourcc(*'mp4v')
        rgb_frame = self.env.render("rgb_array")
        real_shape = (int(rgb_frame.shape[1]), int(rgb_frame.shape[0]))
        video = VideoWriter("./{}.mp4".format(filename), fourcc, float(self.FPS), real_shape)

        for _ in range(self.FPS*self.seconds):
            rgb_frame = self.env.render("rgb_array")
            rgb_frame = cv2.resize(
                rgb_frame, real_shape,
                interpolation=cv2.INTER_AREA)

            t_state = torch.cuda.FloatTensor(self.state)
            dist, self.hidden, value = self.nnet.forward(t_state, self.hidden)
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
            dist = self.nnet.wrap_dist(dist)
            action = dist.sample().detach()
            log_prob = dist.log_prob(action)
            loss = (torch.exp(log_prob) * 0.1).mean()
            loss.backward()

            self.output = self.outputs[-1].detach().cpu().numpy()
            self.gradient = self.gradients[-1].detach().cpu().numpy()[0]

            cam = numpy.zeros(self.output.shape[1:], dtype=numpy.float32)
            for i, w in enumerate(self.gradient):
                cam += w * self.output[i, :, :]

            cam = cam - numpy.mean(cam)

            cam = numpy.maximum(cam, 0)
            cam = cv2.resize(cam, real_shape)
            cam = cam - numpy.min(cam)
            cam = cam / numpy.max(cam)
            cam = cam * 255.
            cam = cam.astype(numpy.uint8)

            cam: numpy.ndarray = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

            cam[:, :, 0] -= numpy.sum(cam, axis=-1).min()

            transparency = 0.7
            cam = cam * transparency
            cam = cam.astype(numpy.uint8)

            cam = numpy.clip(cam, 0, 255)

            cam = cv2.resize(
                cam, real_shape,
                interpolation=cv2.INTER_AREA)

            res = cv2.add(rgb_frame, cam)

            prep = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            prep = cv2.resize(
                prep, real_shape, interpolation=cv2.INTER_AREA
            )

            video.write(prep)

            self.state, _, done, _ = self.env.step(dist.sample().cpu().numpy())

            if done.item():
                self.hidden = None

        video.release()

    def _save_output(self, module, input, output):
        self.outputs.append(output[0])

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])


env_name = "BreakoutNoFrameskip-v4"

env = agnes.make_env(env_name, config={"frame_stack": True})
config, _ = agnes.PPO.get_config(env["env_type"])

runner = agnes.Single(env, agnes.PPO, agnes.LSTMCNN, config=config)

runner.trainer.load("results/Atari-BreakoutNoFrameskip-v4-PPO-10M/Breakout.pth")

VisualizeAttention(env, runner, seconds=60, layer_num=1).run()

print("Done!")
