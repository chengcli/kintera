#!/usr/bin/env python3
"""Build a standalone PDF from the reaction RST chapters and algorithm pages.

Requires docs/requirements.txt, beautifulsoup4, weasyprint, Node.js, and
mathjax-full@3.2.2 (available through NODE_PATH when installed outside docs).
Example: python docs/build_reaction_report.py
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

from bs4 import BeautifulSoup
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
CSS = '''
@page { size: A4; margin: 20mm 17mm 19mm;
  @top-left { content: "KINTERA · Reaction data and equilibrium validation";
              font: 8pt sans-serif; color: #536475; }
  @bottom-right { content: counter(page) " / " counter(pages);
                  font: 8pt sans-serif; color: #536475; }
}
body { font-family: "DejaVu Sans", sans-serif; font-size: 9pt;
       line-height: 1.45; color: #192c3c; }
h1 { font-size: 25pt; line-height: 1.2; color: #123e62; margin: 10mm 0; }
h2 { font-size: 19pt; color: #123e62; break-before: page; }
h3 { font-size: 14pt; color: #123e62; }
h4 { font-size: 11pt; color: #123e62; }
h1,h2,h3,h4,h5 { break-after: avoid; }
p { orphans: 3; widows: 3; }
a { color: #1c6089; text-decoration: none; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 7.5pt;
      background: #f0f4f7; padding: 7pt; border-left: 2pt solid #6b8fa7; }
code { font-size: 8pt; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; margin: 10pt 0;
        font-size: 6.8pt; line-height: 1.3; table-layout: fixed; }
thead { display: table-header-group; }
th { background: #e4eef5; text-align: left; }
th,td { padding: 5pt; border: .4pt solid #b3c3cf;
        overflow-wrap: anywhere; vertical-align: top; }
td p,th p { margin: 0; }
tr { break-inside: avoid; }
figure { margin: 12pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 205mm; width: auto; height: auto; }
figcaption { font-size: 8pt; text-align: left; color: #43596b; }
.math.display { text-align: center; margin: 12pt 0; break-inside: avoid; }
.math img { max-width: 100%; }
.math.inline img { vertical-align: middle; }
.admonition { border-left: 2pt solid #6b8fa7; padding: 6pt 10pt; background: #f0f4f7; }
.toc { padding: 10pt 16pt; background: #f0f4f7; }
.toc a::after { content: leader('.') target-counter(attr(href), page); }
.toc li { margin: 4pt 0; }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=HERE/'_build/reports/kintera_reaction_report.pdf')
    args = parser.parse_args()
    build = HERE/'_build/report-html'
    env = dict(os.environ, KINTERA_DOCS_OFFLINE='1')
    excluded = 'index.rst,api_reference.rst,examples.rst,installation.rst,quickstart.rst,user_guide.rst'
    subprocess.run([sys.executable, '-m', 'sphinx', '-W', '--keep-going',
                    '-b', 'singlehtml', '-D', 'root_doc=reaction_report',
                    '-D', 'exclude_patterns='+excluded,
                    str(HERE/'source'), str(build)], env=env, check=True)
    soup = BeautifulSoup((build/'reaction_report.html').read_text(), 'html.parser')
    main = soup.select_one('[role=main]')
    for item in main.select('.headerlink, script, .viewcode-link'):
        item.decompose()
    equations = main.select('.math')
    inputs = []
    for eq in equations:
        raw = eq.get_text().strip()
        display = eq.name == 'div'
        assert raw.startswith('\\[' if display else '\\('), raw
        inputs.append(dict(tex=raw[2:-2], display=display))
    rendered = json.loads(subprocess.check_output(
        ['node', str(HERE/'render_report_math.cjs')], input=json.dumps(inputs), text=True))
    for eq, svg, spec in zip(equations, rendered, inputs):
        image = soup.new_tag('img')
        image['src'] = 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()
        image['alt'] = spec['tex']
        eq.clear(); eq.append(image)
        eq['class'] = ['math', 'display' if spec['display'] else 'inline']
    for link in main.select('a[href]'):
        href = link['href']
        if href.startswith('reaction_report.html#'):
            link['href'] = href.split('.html', 1)[1]
        elif '_downloads/' in href:
            del link['href']
            link['title'] = 'Supplementary data available in the source repository'
    toc = soup.new_tag('div', attrs={'class': 'toc'})
    label = soup.new_tag('strong'); label.string = 'Contents'; toc.append(label)
    listing = soup.new_tag('ul'); toc.append(listing)
    # Every included document is a section with a stable anchor.
    for section in main.select('section'):
        heading = section.find(['h2', 'h3'], recursive=False)
        if heading and (heading.name == 'h2' or section.get('id') in (
                'water','ammonia','hydrogen-sulfide','methane','sulfur-dioxide',
                'carbon-dioxide','potassium-chloride','ammonium-hydrosulfide',
                'potassium-and-hydrogen-chloride','mns-formation','zns-formation',
                'na2s-formation','magnesium-silicate')):
            if heading.name == 'h3':
                heading.name = 'h2'  # Each reaction starts on a fresh page.
            item = soup.new_tag('li'); anchor = soup.new_tag('a', href='#'+section['id'])
            anchor.string = heading.get_text(); item.append(anchor); listing.append(item)
    main.select_one('h1').insert_after(toc)
    title = 'Kintera — Reaction Data and Equilibrium Validation'
    document = '<!doctype html><html><head><meta charset="utf-8"><title>'+title+'</title><style>'+CSS+'</style></head><body>'+str(main)+'</body></html>'
    (build/'print.html').write_text(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = HTML(string=document, base_url=str(build)+'/').render()
    report.write_pdf(args.output)
    print(f'Wrote {args.output}: {len(report.pages)} pages, {len(equations)} equations, '
          f'{len(main.select("figure"))} figure placements.')


if __name__ == '__main__':
    main()
