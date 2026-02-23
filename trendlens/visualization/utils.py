def ensure_backend():
    import matplotlib
    if matplotlib.get_backend().lower() == 'agg':
        #this is the helper function AI found for compatibility!
        for backend in ('macosx', 'TkAgg', 'Qt5Agg'):
            try:
                matplotlib.use(backend, force=True)
                return
            except Exception:
                continue






