"""Build the website (GitHub Pages) into _site/: the pages in site/, the guides in docs/*.md turned into
HTML, the preset gallery, and the media in docs/media.

    python tools/build_site.py            # needs only Python-Markdown (pip install markdown)
    python tools/build_site.py --presets  # first refresh docs/presets.json from the package (needs the app's
                                          # own environment, for numpy)
    python tools/build_site.py --serve    # build, then serve _site on http://localhost:8000

The guides are plain Markdown and read fine on GitHub too. On the site, an image of an animated GIF in
docs/media/gif/NAME.gif is shown as the sharper video docs/media/video/NAME.mp4 when there is one, and an
image's title ("...") becomes its caption.
"""
import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE, DOCS, OUT = ROOT / 'site', ROOT / 'docs', ROOT / '_site'
REPO = 'https://github.com/gjnail/blackbody'
BASE_URL = 'https://gjnail.github.io/blackbody/'

# (group, slug, short title for the menus). The page title is the guide's own first heading.
GUIDES = [
    ('Start here', 'getting-started', 'Install and first steps'),
    ('Start here', 'tutorial', 'Tutorial: your first shot'),
    ('Fire', 'fire', 'Fire, smoke and sparks'),
    ('Fire', 'fabric', 'Fabric and burning cloth'),
    ('Things', 'physics', 'Things that fall'),
    ('Things', 'matter', 'Sand, snow and mud'),
    ('Things', 'grass', 'Grass and plants'),
    ('Water', 'liquids', 'Liquids'),
    ('Water', 'ocean', 'The sea, surf and rivers'),
    ('Water', 'lava', 'Lava, and fire with water'),
    ('Water', 'heat', 'Ice, boiling and steam'),
    ('Sky', 'weather', 'Weather and clouds'),
    ('Your shot', 'compositing', 'Fitting it into your footage'),
    ('Your shot', 'scene-import', 'Moving shots and scene import'),
    ('Your shot', 'outputs', 'Outputs'),
    ('Your shot', 'command-line', 'Command line and farms'),
    ('Reference', 'performance', 'Performance and memory'),
    ('Reference', 'how-it-works', 'How it works'),
    ('Reference', 'troubleshooting', 'Troubleshooting and limits'),
]
GUIDE_SLUGS = {g[1] for g in GUIDES}

# the gallery's filters: preset category -> filter
FILTERS = [('all', 'All'), ('fire', 'Fire and smoke'), ('things', 'Things that fall'), ('matter', 'Sand, snow and mud'),
           ('water', 'Water'), ('sea', 'The sea'),
           ('heat', 'Ice and steam'), ('weather', 'Weather and sky')]
CATEGORY_FILTER = {'Fires': 'fire', 'Small flames': 'fire', 'Explosions': 'fire', 'Smoke': 'fire', 'Sparks': 'fire',
                   'Steam': 'fire', 'Liquids': 'water', 'Fire and liquid': 'water', 'Sea': 'sea',
                   'Ice and steam': 'heat', 'Weather': 'weather', 'Sky and weather': 'weather', 'Things that fall': 'things',
                   'Sand, snow and mud': 'matter'}

NAV = [('tutorial', 'Tutorial', 'tutorial.html'), ('guides', 'Guides', 'guides.html'),
       ('presets', 'Presets', 'presets.html'), ('github', 'GitHub', REPO)]


def esc(s):
    return html.escape(str(s), quote=True)


# ---- page shell -------------------------------------------------------------------------------------------

def head(title, description, path, image='media/img/og.jpg'):
    full = title if title.startswith('Blackbody') else f'{title} · Blackbody'
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(full)}</title>
  <meta name="description" content="{esc(description)}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{esc(full)}">
  <meta property="og:description" content="{esc(description)}">
  <meta property="og:image" content="{BASE_URL}{image}">
  <meta property="og:url" content="{BASE_URL}{path}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="theme-color" content="#0a0a0c">
  <link rel="icon" href="media/img/icon-64.png" type="image/png">
  <link rel="preload" href="assets/fonts/barlow-700.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>'''


def header(current):
    links = []
    for key, label, href in NAV:
        cur = ' aria-current="page"' if key == current else ''
        links.append(f'<a href="{href}"{cur}>{label}</a>')
    links.append('<a class="btn btn-primary btn-small" href="getting-started.html">Get Blackbody</a>')
    return f'''<header class="site-header">
  <div class="header-inner">
    <a class="brand" href="index.html"><img src="media/img/icon-64.png" alt="" width="28" height="28">Blackbody</a>
    <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="site-nav">Menu</button>
    <nav class="site-nav" id="site-nav" aria-label="Main">
      {"".join(links)}
    </nav>
  </div>
</header>'''


def footer():
    return f'''<footer class="site-footer">
  <div class="footer-inner">
    <div>Blackbody · GPU fire, smoke and water for live-action footage · MIT license</div>
    <nav aria-label="Footer">
      <a href="getting-started.html">Install</a>
      <a href="guides.html">Guides</a>
      <a href="presets.html">Presets</a>
      <a href="{REPO}">Source</a>
      <a href="{REPO}/blob/main/CHANGELOG.md">Changelog</a>
      <a href="{REPO}/issues">Report a problem</a>
      <a href="https://ko-fi.com/gnail">Support on Ko-fi</a>
    </nav>
  </div>
</footer>
<script src="assets/site.js" defer></script>
</body>
</html>
'''


# ---- media helpers -------------------------------------------------------------------------------------------

def video_tag(name, caption=None, label=None, cls='media', controls=False):
    """A looping, muted clip that plays while on screen (media/video/NAME.mp4 with its poster)."""
    v = f'<video data-loop muted loop playsinline preload="none" poster="media/video/{name}.jpg"' \
        f'{" controls" if controls else ""}><source src="media/video/{name}.mp4" type="video/mp4"></video>'
    lab = f'<span class="label">{esc(label)}</span>' if label else ''
    cap = f'<figcaption>{caption}</figcaption>' if caption else ''
    return f'<figure><div class="{cls}">{v}{lab}</div>{cap}</figure>'


IMG_P = re.compile(r'<p>\s*((?:<img [^>]*>\s*)+)</p>')
IMG = re.compile(r'<img [^>]*>')
IMG_ATTR = re.compile(r'(\w+)="([^"]*)"')


def figure(img_tag):
    """One image -> a figure: a GIF that has a video becomes the video; its title becomes the caption."""
    attrs = dict(IMG_ATTR.findall(img_tag))
    src, alt, title = attrs.get('src', ''), attrs.get('alt', ''), attrs.get('title')
    cap = f'<figcaption>{title}</figcaption>' if title else ''
    g = re.fullmatch(r'media/gif/([\w-]+)\.gif', src)
    if g and (DOCS / 'media' / 'video' / f'{g.group(1)}.mp4').exists():
        name = g.group(1)
        return (f'<figure><video data-loop muted loop playsinline preload="none" aria-label="{alt}" '
                f'poster="media/video/{name}.jpg"><source src="media/video/{name}.mp4" type="video/mp4"></video>'
                f'{cap}</figure>')
    return f'<figure><img src="{src}" alt="{alt}" loading="lazy">{cap}</figure>'


def figures(body):
    """<p><img></p> -> a figure; several images in one paragraph (side by side on GitHub) -> a grid."""
    def fig(m):
        imgs = IMG.findall(m.group(1))
        if len(imgs) == 1:
            return figure(imgs[0])
        return '<div class="grid2">' + ''.join(figure(t) for t in imgs) + '</div>'
    return IMG_P.sub(fig, body)


def links(body):
    """Markdown links between the docs and to the source -> site links."""
    def fix(m):
        href = m.group(1)
        if re.match(r'^[a-z]+:', href) or href.startswith('#'):
            return m.group(0)
        path, _, frag = href.partition('#')
        frag = f'#{frag}' if frag else ''
        name = Path(path).name
        if path.endswith('.md') and Path(path).stem in GUIDE_SLUGS and '/' not in path.strip('./'):
            return f'href="{Path(path).stem}.html{frag}"'
        if path in ('../README.md', 'README.md'):
            return f'href="{REPO}#readme"'
        if path.startswith('../'):
            kind = 'tree' if path.endswith('/') else 'blob'
            return f'href="{REPO}/{kind}/main/{path[3:]}{frag}"'
        if name == 'presets.md':
            return f'href="presets.html{frag}"'
        return m.group(0)
    return re.sub(r'href="([^"]+)"', fix, body)


# ---- guides ----------------------------------------------------------------------------------------------------

def render_md(text):
    import markdown
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'attr_list', 'md_in_html', 'sane_lists', 'toc'],
                           extension_configs={'toc': {'permalink': '#', 'toc_depth': '2-3'}})
    body = md.convert(text)
    return body, md.toc_tokens


def doc_nav(current):
    out, group = [], None
    for g, slug, short in GUIDES:
        if g != group:
            out.append(f'<h4>{esc(g)}</h4>')
            group = g
        cur = ' aria-current="page"' if slug == current else ''
        out.append(f'<a href="{slug}.html"{cur}>{esc(short)}</a>')
    return ('<aside class="doc-nav" aria-label="Guides"><button class="doc-nav-toggle" type="button" '
            'aria-expanded="false">All guides</button><div class="doc-nav-list">' + ''.join(out) + '</div></aside>')


def toc_html(tokens):
    def walk(ts):
        items = []
        for t in ts:
            kids = walk(t['children']) if t['children'] else ''
            items.append(f'<li><a href="#{t["id"]}">{t["name"]}</a>{kids}</li>')
        return f'<ul>{"".join(items)}</ul>' if items else ''
    h2 = tokens[0]['children'] if tokens and tokens[0]['level'] == 1 else tokens
    if not h2:
        return ''
    flat = [dict(t, children=[c for c in t['children'] if c['level'] == 3]) for t in h2]
    return f'<nav class="toc" aria-label="On this page"><h4>On this page</h4>{walk(flat)}</nav>'


def build_guide(i, slug):
    src = DOCS / f'{slug}.md'
    text = src.read_text(encoding='utf-8')
    body, tokens = render_md(text)
    m = re.search(r'<h1[^>]*>(.*?)</h1>', body, re.S)
    title = re.sub(r'<[^>]+>', '', m.group(1)).replace('#', '').strip() if m else slug
    # the first paragraph after the title is the page's introduction
    body = re.sub(r'(</h1>\s*)<p>', r'\1<p class="intro">', body, count=1)
    intro = re.search(r'<p class="intro">(.*?)</p>', body, re.S)
    desc = re.sub(r'<[^>]+>', '', intro.group(1)).strip() if intro else title
    desc = (desc[:240] + '…') if len(desc) > 240 else desc
    body = links(figures(body))
    prev_g = GUIDES[i - 1] if i > 0 else None
    next_g = GUIDES[i + 1] if i + 1 < len(GUIDES) else None
    pager = '<nav class="pager" aria-label="Previous and next">'
    pager += (f'<a class="prev" href="{prev_g[1]}.html"><small>Previous</small><b>{esc(prev_g[2])}</b></a>'
              if prev_g else '<span></span>')
    pager += (f'<a class="next" href="{next_g[1]}.html"><small>Next</small><b>{esc(next_g[2])}</b></a>'
              if next_g else '<span></span>')
    pager += '</nav>'
    page = (head(title, desc, f'{slug}.html') + header('tutorial' if slug == 'tutorial' else 'guides') +
            f'<main id="main" class="doc">{doc_nav(slug)}<article class="prose">{body}'
            f'<p class="caption">Something wrong or missing on this page? <a href="{REPO}/edit/main/docs/{slug}.md">'
            f'Edit it on GitHub</a>.</p>{pager}</article>{toc_html(tokens)}</main>' + footer())
    (OUT / f'{slug}.html').write_text(page, encoding='utf-8')
    return title, desc


def build_guides_index(info):
    cards, group = [], None
    blurb_media = json.loads((SITE / 'guides.json').read_text(encoding='utf-8'))
    for g, slug, short in GUIDES:
        if g != group:
            if group is not None:
                cards.append('</div>')
            cards.append(f'<h2 class="group-title">{esc(g)}</h2><div class="cards">')
            group = g
        title, desc = info[slug]
        thumb = blurb_media.get(slug, {})
        img = thumb.get('image', 'media/img/og.jpg')
        text = thumb.get('blurb', desc)
        cards.append(f'<a class="card" href="{slug}.html"><img src="{img}" alt="" loading="lazy" width="640" height="360">'
                     f'<div class="card-body"><h3>{esc(short)}</h3><p>{esc(text)}</p></div></a>')
    cards.append('</div>')
    page = (head('Guides', 'Every part of Blackbody explained: fire, smoke, liquids, the sea, lava, ice, weather, '
                 'compositing into footage, outputs and the command line.', 'guides.html') + header('guides') +
            '<main id="main"><section class="section"><div class="section-head"><p class="kicker">Guides</p>'
            '<h1 style="font:700 clamp(36px,5vw,52px)/1.05 var(--head);margin:0 0 14px">Everything Blackbody does, '
            'and how to use it</h1><p class="sub">Start with the install guide and the tutorial. Every other '
            'guide stands on its own: read the one for the effect you need.</p></div>' + ''.join(cards) +
            '</section></main>' + footer())
    page = page.replace('<h2 class="group-title">', '<h2 class="group-title" style="font:700 14px/1 var(--head);'
                        'letter-spacing:.24em;text-transform:uppercase;color:var(--faint);margin:44px 0 16px">')
    (OUT / 'guides.html').write_text(page, encoding='utf-8')


# ---- presets ---------------------------------------------------------------------------------------------------

def refresh_presets():
    sys.path.insert(0, str(ROOT))
    from blackbody.scene import presets
    rows = []
    for key in presets.ORDER:
        p = presets.PRESETS[key]
        rows.append({'key': key, 'name': p['name'], 'category': p['category'], 'size': p['size'], 'blurb': p['blurb']})
    (DOCS / 'presets.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    # the same catalogue as Markdown, for reading on GitHub (the site builds presets.html from the JSON)
    md = ['# Presets', '',
          f'Blackbody comes with {len(rows)} presets. Click one in the app\'s Effects panel to load it into your shot; '
          'every one is an ordinary scene you can change and save as your own. On the website the gallery plays '
          'a clip of each one that has one: <https://gjnail.github.io/blackbody/presets.html>.', '']
    for f_key, f_label in FILTERS[1:]:
        group = [r for r in rows if CATEGORY_FILTER.get(r['category'], 'fire') == f_key]
        if not group:
            continue
        md += [f'## {f_label}', '', '| | Preset | Notes |', '|---|---|---|']
        for r in group:
            blurb = r['blurb'].replace('|', '\\|')
            md.append(f'| <img src="../blackbody/assets/presets/{r["key"]}.png" width="200" alt=""> | '
                      f'**{r["name"]}**<br>{r["size"]} | {blurb} |')
        md.append('')
    (DOCS / 'presets.md').write_text('\n'.join(md), encoding='utf-8', newline='\n')
    print('presets.json and presets.md:', len(rows), 'presets')


def build_presets():
    rows = json.loads((DOCS / 'presets.json').read_text(encoding='utf-8'))
    thumbs = OUT / 'media' / 'presets'
    thumbs.mkdir(parents=True, exist_ok=True)
    chips = ''.join(f'<button type="button" data-cat="{k}" aria-pressed="{"true" if k == "all" else "false"}">'
                    f'{esc(v)}</button>' for k, v in FILTERS)
    cards = []
    for r in rows:
        key = r['key']
        src = ROOT / 'blackbody' / 'assets' / 'presets' / f'{key}.png'
        if src.exists():
            shutil.copy2(src, thumbs / f'{key}.png')
        cat = CATEGORY_FILTER.get(r['category'], 'fire')
        has_video = (DOCS / 'media' / 'video' / f'{key}.mp4').exists()
        vid = (f'<video muted loop playsinline preload="none"><source src="media/video/{key}.mp4" type="video/mp4"></video>'
               '<span class="play">Video</span>') if has_video else ''
        water = ' water' if cat in ('water', 'sea', 'heat') else ''
        cards.append(f'<article class="preset" data-cat="{cat}" id="{key}" tabindex="0"><div class="thumb">'
                     f'<img src="media/presets/{key}.png" alt="" loading="lazy" width="640" height="360">{vid}</div>'
                     f'<div class="body"><h3>{esc(r["name"])}</h3><p class="meta{water}">{esc(r["category"])} · '
                     f'{esc(r["size"])}</p><p>{esc(r["blurb"])}</p></div></article>')
    page = (head('Presets', f'All {len(rows)} built-in Blackbody presets: fires, explosions, sparks, steam, water, '
                 'the sea, lava, ice, weather and clouds.', 'presets.html') + header('presets') +
            '<main id="main"><section class="section" style="padding-top:56px"><div class="section-head">'
            '<p class="kicker">Library</p><h1 style="font:700 clamp(36px,5vw,52px)/1.05 var(--head);margin:0 0 14px">'
            f'{len(rows)} presets to start from</h1><p class="sub">Click one in the app\'s Effects panel to load it '
            'into your shot. Every one is an ordinary scene: change anything, then save it as your own preset. '
            'Hover a card marked Video to watch it.</p></div>'
            f'<div class="filters" role="group" aria-label="Filter presets">{chips}</div>'
            f'<div class="preset-grid">{"".join(cards)}</div></section></main>' + footer())
    (OUT / 'presets.html').write_text(page, encoding='utf-8')


# ---- static pages ---------------------------------------------------------------------------------------------

def build_static():
    """site/*.html: {{head|TITLE|DESCRIPTION|PATH}}, {{header|KEY}}, {{footer}} and {{video|NAME|CAPTION|LABEL}}."""
    count = len(json.loads((DOCS / 'presets.json').read_text(encoding='utf-8')))
    # the building blocks in Create, counted in the source (the site builds without the package's dependencies)
    blocks = len(re.findall(r"^    Component\('", (ROOT / 'blackbody' / 'scene' / 'components.py').read_text(encoding='utf-8'), re.M))
    for src in SITE.glob('*.html'):
        text = src.read_text(encoding='utf-8').replace('{{preset_count}}', str(count)).replace('{{block_count}}', str(blocks))
        text = re.sub(r'\{\{head\|([^|]*)\|([^|]*)\|([^}]*)\}\}', lambda m: head(m.group(1), m.group(2), m.group(3)), text)
        text = re.sub(r'\{\{header\|([^}]*)\}\}', lambda m: header(m.group(1)), text)
        text = text.replace('{{footer}}', footer())
        text = re.sub(r'\{\{video\|([^|}]*)\|?([^|}]*)\|?([^}]*)\}\}',
                      lambda m: video_tag(m.group(1), m.group(2) or None, m.group(3) or None), text)
        if src.name == '404.html':   # served for any missing path, so its links must not be relative to it
            text = text.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n  <base href="/blackbody/">', 1)
        (OUT / src.name).write_text(text, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--presets', action='store_true', help='refresh docs/presets.json from the package first')
    ap.add_argument('--serve', action='store_true', help='serve _site on http://localhost:8000 after building')
    args = ap.parse_args()
    if args.presets:
        refresh_presets()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copytree(SITE / 'assets', OUT / 'assets')
    shutil.copytree(DOCS / 'media', OUT / 'media')
    if (DOCS / 'tutorial').exists():
        shutil.copytree(DOCS / 'tutorial', OUT / 'tutorial')
    (OUT / '.nojekyll').write_text('')
    build_static()
    info = {}
    for i, (_, slug, _) in enumerate(GUIDES):
        info[slug] = build_guide(i, slug)
    build_guides_index(info)
    build_presets()
    pages = sorted(p.name for p in OUT.glob('*.html'))
    with open(OUT / 'sitemap.xml', 'w', encoding='utf-8') as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for p in pages:
            if p != '404.html':
                fh.write(f'  <url><loc>{BASE_URL}{"" if p == "index.html" else p}</loc></url>\n')
        fh.write('</urlset>\n')
    print(f'built {len(pages)} pages into {OUT}')
    if args.serve:
        import functools
        import http.server
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        print('serving on http://localhost:8000')
        http.server.ThreadingHTTPServer(('127.0.0.1', 8000), handler).serve_forever()


if __name__ == '__main__':
    main()
