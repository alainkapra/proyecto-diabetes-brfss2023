"""Genera la lista de referencias APA 7 (thebibliography) desde referencias.bib.
Uso: python generar_referencias.py referencias.bib > referencias_apa.tex  (y pegarla al final de main.tex)."""
import re, sys, bibtexparser
from bibtexparser.bparser import BibTexParser
p = BibTexParser(common_strings=True); p.ignore_nonstandard_types = False
db = bibtexparser.load(open(sys.argv[1]), parser=p)

def strip_braces(s): return s.replace('{', '').replace('}', '')
def autores(campo):
    out = []
    nivel=0; trozos=['']
    for tok in re.split(r'(\s+and\s+|[{}])', campo):
        if tok=='{': nivel+=1
        elif tok=='}': nivel-=1
        if tok and re.fullmatch(r'\s+and\s+', tok) and nivel==0: trozos.append('')
        else: trozos[-1]+=tok
    for a in [x.strip() for x in trozos]:
        if a == 'others': out.append(None); continue
        if a.startswith('{') and a.endswith('}'): out.append((a[1:-1], None)); continue
        partes = a.split(' ')
        # apellido = último token, o bloque entre llaves
        m = re.search(r'\{([^}]*)\}$', a)
        if m: ape = m.group(1); ini = a[:m.start()].strip()
        else: ape = partes[-1]; ini = ' '.join(partes[:-1])
        out.append((ape, ini))
    return out
def fmt_lista(lst):
    items = []; etal = False
    for x in lst:
        if x is None: etal = True; continue
        ape, ini = x
        items.append(ape if ini is None else f"{ape}, {ini}")
    if etal: return ', '.join(items) + ', et~al.'
    if len(items) == 1: return items[0]
    return ', '.join(items[:-1]) + ' y ' + items[-1]
def fmt_editores(campo):
    eds = autores(campo)
    nombres = [f"{ini} {ape}" for ape, ini in eds]
    s = nombres[0] if len(nombres) == 1 else ', '.join(nombres[:-1]) + ' y ' + nombres[-1]
    return s + (' (Ed.)' if len(nombres) == 1 else ' (Eds.)')
def etiqueta(lst):
    reales = [x for x in lst if x is not None]
    if len(reales) >= 3 or None in lst: return reales[0][0] + ' et~al.'
    if len(reales) == 2: return f"{reales[0][0]} y {reales[1][0]}"
    return reales[0][0]
def pags(pg): return pg.replace('--', '–') if pg else ''
def doi(e):
    if e.get('doi'): return f" \\url{{https://doi.org/{e['doi']}}}"
    if e.get('url'): return f" \\url{{{e['url']}}}"
    return ''
def punto(t): t = t.strip(); return t if t[-1] in '.?!' else t + '.'

ent = []
for e in db.entries:
    au = autores(e['author']); y = e['year']; t = e['title']; tipo = e['ENTRYTYPE']
    A = fmt_lista(au)
    if tipo == 'article':
        vol = f", \\textit{{{e['volume']}}}" if e.get('volume') else ''
        num = f"({e['number']})" if e.get('number') else ''
        pg = f", {pags(e.get('pages'))}" if e.get('pages') else ''
        cuerpo = f"{punto(t)} \\textit{{{e['journal']}}}{vol}{num}{pg}.{doi(e)}"
    elif tipo == 'book':
        ed = f" ({e['edition']}.ª ed.)" if e.get('edition') else ''
        cuerpo = f"\\textit{{{t}}}{ed}. {e['publisher']}.{doi(e)}"
    elif tipo == 'inproceedings':
        det = []
        if e.get('volume'): det.append(f"Vol.~{e['volume']}")
        if e.get('pages'): det.append(f"pp.~{pags(e['pages'])}")
        d = f" ({', '.join(det)})" if det else ''
        cuerpo = f"{punto(t)} En \\textit{{{e['booktitle']}}}{d}.{doi(e)}"
    elif tipo == 'incollection':
        det = []
        if e.get('volume'): det.append(f"Vol.~{e['volume']}")
        if e.get('pages'): det.append(f"pp.~{pags(e['pages'])}")
        d = f" ({', '.join(det)})" if det else ''
        cuerpo = f"{punto(t)} En {fmt_editores(e['editor'])}, \\textit{{{e['booktitle']}}}{d}. {e['publisher']}.{doi(e)}"
    elif tipo == 'dataset':
        cuerpo = f"\\textit{{{t}}} [Conjunto de datos]. {e['publisher']}.{doi(e)}"
    else:
        raise SystemExit(tipo)
    orden = strip_braces(au[0][0]).lower() + y
    sep = '. ' if au[0] is not None and au[0][1] is None and len(au)==1 else ' '
    ent.append((orden, f"\\bibitem[{{{etiqueta(au)}}}({y})]{{{e['ID']}}}\n{A}{sep}({y}). {cuerpo}\n"))
ent.sort()
print("\\begin{thebibliography}{}")
for _, s in ent: print(s)
print("\\end{thebibliography}")
