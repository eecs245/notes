// Compare the browser inference code against independently computed ONNX results.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'assets/mnist-widget.html'), 'utf8');
const encoded = JSON.parse(html.match(/<script id="mnist-weights" type="application\/json">(.*?)<\/script>/s)[1]);
const weights = Object.fromEntries(Object.entries(encoded).map(([key, value]) => {
  const bytes = Buffer.from(value, 'base64');
  return [key, Float32Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4))];
}));
const context = vm.createContext({weights});
vm.runInContext(html.split('// BEGIN INFERENCE')[1].split('\n').slice(1).join('\n').split('// END INFERENCE')[0], context);
const fixtures = JSON.parse(fs.readFileSync(path.join(root, 'scripts/fixtures/mnist.json')));
for (const fixture of fixtures) {
  const input = Float32Array.from(Buffer.from(fixture.pixels, 'base64'), value => value / 255);
  const probabilities = context.predict(input);
  assert.equal(probabilities.length, 10);
  assert.ok(probabilities.every(p => Number.isFinite(p) && p >= 0 && p <= 1));
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.equal(probabilities.indexOf(Math.max(...probabilities)), fixture.label);
  probabilities.forEach((p, i) => assert.ok(Math.abs(p - fixture.probabilities[i]) < 1e-5));
}
const notebook = JSON.parse(fs.readFileSync(path.join(root, '01_introduction_to_supervised_learning/01-what-is-machine-learning.ipynb')));
const output = notebook.cells[1].outputs[0].data['text/html'].join('');
const escaped = html.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#x27;');
assert.ok(output.includes(`srcdoc="${escaped}"`), 'Saved notebook output must match the widget source');
assert.ok(!output.includes('mco-mnist-draw'));
console.log('Passed: all ten digits match ONNX reference probabilities; saved notebook embeds the current widget.');
