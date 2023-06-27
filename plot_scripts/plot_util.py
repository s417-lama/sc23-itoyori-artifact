import os
import re
import scipy as sp
import scipy.stats
import pandas as pd

def infer_col_dtype(col):
    """
    See https://stackoverflow.com/questions/35003138/python-pandas-inferring-column-datatypes/48269724
    """
    if col.dtype == "object":
        try:
            return pd.to_numeric(col.dropna().unique()).dtype
        except:
            try:
                return pd.to_datetime(col.dropna().unique(), format="dateutil").dtype
            except:
                try:
                    return pd.to_timedelta(col.dropna().unique()).dtype
                except:
                    return "object"
    else:
        return col.dtype

def infer_dtypes(df):
    dtype_dict = {col: infer_col_dtype(df[col]) for col in df.columns}
    return df.astype(dtype=dtype_dict)

def confidence_interval(data):
    return sp.stats.t.interval(confidence=0.95,
                               loc=data.mean(),
                               df=len(data) - 1,
                               scale=sp.stats.sem(data))

def ci_lower(data):
    return confidence_interval(data)[0]

def ci_upper(data):
    return confidence_interval(data)[1]

def txt2df(files, ctx_regex_patterns, row_regex_patterns):
    ctx_regexes = [re.compile(r) for r in ctx_regex_patterns]
    row_regexes = [re.compile(r) for r in row_regex_patterns]
    rows = []
    for file in files:
        if isinstance(file, tuple):
            filename, ctx = file
        else:
            filename = file
            ctx = dict()
        try:
            with open(filename, "r") as f:
                for line in f:
                    for r in ctx_regexes:
                        m = r.search(line)
                        if m:
                            ctx.update(m.groupdict())
                    for r in row_regexes:
                        m = r.search(line)
                        if m:
                            rows.append(dict(ctx, **m.groupdict()))
        except FileNotFoundError:
            print("Warning: File '{}' not found.".format(filename))
    return infer_dtypes(pd.DataFrame(rows))

def save_fig(fig, fig_dir, filename):
    os.makedirs(fig_dir, exist_ok=True)
    savepath = os.path.join(fig_dir, filename)
    print("The plot was saved to {}".format(savepath))
    fig.write_html(savepath, include_plotlyjs="cdn", config={"toImageButtonOptions" : {"format" : "svg"}})

"""
Definition of colour schemes for lines and maps that also work for colour-blind
people. See https://personal.sron.nl/~pault/ for background information and
best usage of the schemes.

Copyright (c) 2022, Paul Tol
All rights reserved.

License:  Standard 3-clause BSD
"""
def tol_cset(colorset=None):
    """
    Discrete color sets for qualitative data.

    Define a namedtuple instance with the colors.
    Examples for: cset = tol_cset(<scheme>)
      - cset.red and cset[1] give the same color (in default 'bright' colorset)
      - cset._fields gives a tuple with all color names
      - list(cset) gives a list with all colors
    """
    from collections import namedtuple

    namelist = ('bright', 'high-contrast', 'vibrant', 'muted', 'medium-contrast', 'light')
    if colorset is None:
        return namelist
    if colorset not in namelist:
        colorset = 'bright'
        print('*** Warning: requested colorset not defined,',
              'known colorsets are {}.'.format(namelist),
              'Using {}.'.format(colorset))

    if colorset == 'bright':
        cset = namedtuple('Bcset',
                    'blue red green yellow cyan purple grey black')
        return cset('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                    '#AA3377', '#BBBBBB', '#000000')

    if colorset == 'high-contrast':
        cset = namedtuple('Hcset',
                    'blue yellow red black')
        return cset('#004488', '#DDAA33', '#BB5566', '#000000')

    if colorset == 'vibrant':
        cset = namedtuple('Vcset',
                    'orange blue cyan magenta red teal grey black')
        return cset('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
                    '#009988', '#BBBBBB', '#000000')

    if colorset == 'muted':
        cset = namedtuple('Mcset',
                    'rose indigo sand green cyan wine teal olive purple pale_grey black')
        return cset('#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                    '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD',
                    '#000000')

    if colorset == 'medium-contrast':
        cset = namedtuple('Mcset',
                    'light_blue dark_blue light_yellow dark_red dark_yellow light_red black')
        return cset('#6699CC', '#004488', '#EECC66', '#994455', '#997700',
                    '#EE99AA', '#000000')

    if colorset == 'light':
        cset = namedtuple('Lcset',
                    'light_blue orange light_yellow pink light_cyan mint pear olive pale_grey black')
        return cset('#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
                    '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#000000')
