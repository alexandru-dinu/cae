import os
import sys
import subprocess

import hug

net_out_img_path = '/tmp/cae_out.png'
out_img_path = None


@hug.post('/upload')
def upload_image(body):
    global out_img_path

    ws = str(body['smoothing_window_size'])

    out = subprocess.check_output([sys.executable, 'test.py', '--api', str(body)])
    out = out.decode('utf-8')
    i = out.index('avg_loss: ')
    loss = out[i:i + 18]

    subprocess.check_output([sys.executable, 'smoothing.py', '--in_img', net_out_img_path, '--window_size', ws])
    f, e = os.path.splitext(net_out_img_path)
    out_img_path = f"{f}_s{ws}{e}"

    return loss


@hug.get('/output', output=hug.output_format.file)
def hello():
    return "" if out_img_path is None else out_img_path
