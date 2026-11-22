"""Ephemeral synthetic CPU/GPU ABI check; no optimizer or saved checkpoints."""
import io


def smoke(device='cpu'):
    import torch
    from .models import LWNetFullAxis, LWNetSpectrum, LWNetSpatial, N2N4MFullAxis
    if device not in ('cpu','cuda') or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested smoke-test device is unavailable')
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        wave=torch.arange(20,dtype=torch.float32,device=device)*6+1000
        detector=torch.zeros(20,dtype=torch.int64,device=device)
        edges=torch.ones(19,dtype=torch.bool,device=device)
        edges[9]=False
        spectral = LWNetSpectrum(2)
        spatial = LWNetSpatial(2)
        spatial.load_spectral_state(spectral.state_dict())
        for name,model,shape in [('n2n4m',N2N4MFullAxis(),(2,20)),
                                 ('lwnet_legacy',LWNetFullAxis(2),(2,1,20,13,13)),
                                 ('lwnet_spectrum',spectral,(1,20)),
                                 ('lwnet_spatial',spatial,(1,1,20,13,13))]:
            model=model.to(device).eval()
            values=torch.ones(shape,device=device)
            valid=torch.ones(shape,dtype=torch.bool,device=device)
            values.reshape(-1)[0]=float('nan');valid.reshape(-1)[0]=False
            output=model(values,valid,wave,detector,edges)
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite synthetic prediction')
            output.square().mean().backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite synthetic gradient')
            buffer=io.BytesIO()
            torch.jit.save(torch.jit.script(model),buffer)
            buffer.seek(0)
            loaded=torch.jit.load(buffer,map_location=device)
            torch.testing.assert_close(output,loaded(values,valid,wave,detector,edges))
            print(f'{name}: masked forward/backward and TorchScript round-trip OK ({device})')
    finally:
        torch.set_num_threads(old_threads)
