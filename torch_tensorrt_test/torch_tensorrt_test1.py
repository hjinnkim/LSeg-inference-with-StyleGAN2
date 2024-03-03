
import torch
from torchvision.models.mobilenet import mobilenet_v2
import tensorrt
import torch_tensorrt


net = mobilenet_v2(pretrained=True).cuda().eval()
torch_script_module = torch.jit.script(net)

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 112, 112],
            opt_shape=[1, 3, 224, 224],
            max_shape=[1, 3, 448, 448],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16
)
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript