def _missing(*_, **__):
    raise NotImplementedError("padding helpers are not shimmed")


index_first_axis = pad_input = unpad_input = _missing
