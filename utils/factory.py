def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    if name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "memo":
        from models.memo import MEMO
        return MEMO(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    else:
        assert 0
