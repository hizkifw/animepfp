import io
import sys
import time
import torch
import numpy as np
import hashlib
import functools
import PIL
from numpy.random import default_rng
from torchvision.transforms import ToPILImage
from gan.models import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer


def load_generator(model_name_or_path):
    generator = Generator(in_channels=256, out_channels=3)
    generator = generator.from_pretrained(
        model_name_or_path, in_channels=256, out_channels=3
    )
    _ = generator.eval()
    return generator


def _denormalize(input: torch.Tensor) -> torch.Tensor:
    return (input * 127.5) + 127.5


# Load generator
generator = load_generator("huggan/fastgan-few-shot-anime-face")


@functools.lru_cache(maxsize=128)
def generate(seed):
    rng = default_rng(seed)
    noise = torch.from_numpy(rng.standard_normal((1, 256, 1, 1)).astype(np.float32))
    with torch.no_grad():
        outputs, _ = generator(noise)

    outputs = _denormalize(outputs.detach())
    image = ToPILImage()(outputs[0].cpu() / 255.0)

    # Scale image to 256x256
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)

    return image


def str_to_seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2 ** 32


class Server(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "image/jpeg")

        t_start = time.time()
        image = generate(str_to_seed(self.path[1:]))
        t_end = time.time()
        t_elapsed = t_end - t_start

        self.send_header("X-Took", str(t_elapsed) + " seconds")
        self.end_headers()

        image.save(self.wfile, format="JPEG")


hostName = "0.0.0.0"
serverPort = 8080

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quit":
        print("Quitting")
        sys.exit(0)

    webServer = HTTPServer((hostName, serverPort), Server)
    print("Server started http://%s:%s" % (hostName, serverPort), flush=True)

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
