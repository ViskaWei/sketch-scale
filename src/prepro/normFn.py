



def get_min_max_norm(df, out=False):
    vmin,vmax=df.min().min(), df.max().max()
    dfNorm=((df-vmin)/(vmax-vmin))
    assert ((dfNorm>=0) & (dfNorm<=1)).all().all()
    if out: 
        return dfNorm, vmin, vmax
    else:
        return dfNorm