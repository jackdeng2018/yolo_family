from .spp import SPPBlock, SPPBlockCSP, SPPBlockDW
from yolo.models.basic import ConvBlocks

def build_neck(model, in_ch, out_ch, act='lrelu'):
    if model == 'conv_blocks':
        print("Neck: ConvBlocks")
        neck = ConvBlocks(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp':
        print("Neck: SPP")
        neck = SPPBlock(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp-csp':
        print("Neck: SPP-CSP")
        neck = SPPBlockCSP(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp-dw':
        print("Neck: SPP-DW")
        neck = SPPBlockDW(c1=in_ch, c2=out_ch, act=act)

    return neck