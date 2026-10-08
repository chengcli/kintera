// Convert Sphinx TeX nodes to self-contained SVG without a browser or CDN.
const fs = require('fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  InputJax: new TeX({packages: AllPackages}),
  OutputJax: new SVG({fontCache: 'none'})
});
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = input.map(({tex, display}) => {
  const node = document.convert(tex, {display, em: 16, ex: 8, containerWidth: 680});
  const svg = adaptor.outerHTML(adaptor.firstChild(node));
  if (svg.includes('data-mml-node="merror"')) throw new Error(`Invalid TeX: ${tex}`);
  return svg;
});
process.stdout.write(JSON.stringify(output));
