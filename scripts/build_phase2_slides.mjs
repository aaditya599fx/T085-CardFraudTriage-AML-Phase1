import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const PptxGenJS = require('pptxgenjs');
const JSZip = require('jszip');
const root = fileURLToPath(new URL('../', import.meta.url));
const study = path.join(root, 'results/phase2/study');
const hybrid = path.join(root, 'results/phase2/hybrid');
const destination = process.argv[2] ?? path.join(root, 'docs/phase2/T085_Phase2.pptx');

async function records(file, folder = study) {
  const text = (await fs.readFile(path.join(folder, file), 'utf8')).trim();
  const [header, ...lines] = text.split(/\r?\n/);
  const keys = header.split(',');
  return lines.map((line) => {
    const quoted = /^"([^"]+)",(.*)$/.exec(line);
    const values = quoted ? [quoted[1], ...quoted[2].split(',')] : line.split(',');
    return Object.fromEntries(keys.map((key, index) => [key, values[index]]));
  });
}

const [ranking, operating, capacity, ablation, hybridRanking, hybridCapacity, hybridOperating, hybridMeta] = await Promise.all([
  records('ranking.csv'), records('operating_points.csv'),
  records('capacity.csv'), records('ablation_summary.csv'),
  records('ranking.csv', hybrid), records('capacity.csv', hybrid), records('operating_points.csv', hybrid),
  fs.readFile(path.join(hybrid, 'metadata.json'), 'utf8').then(JSON.parse),
]);

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Team T085';
pptx.subject = 'Project 26, Phase 2 card fraud triage';
pptx.title = 'Card fraud triage';

const colors = { paper: 'F7F8F5', accent: '147A89', ink: '1B3542', foot: '526C75', white: 'FFFFFF', stripe: 'EAF1EF' };
const percent = (value, digits = 1) => (100 * Number(value)).toFixed(digits) + '%';
const decimal = (value) => Number(value).toFixed(4);

function text(slide, value, x, y, w, h, options = {}) {
  slide.addText(value, { x, y, w, h, fontFace: 'Aptos', fontSize: 22,
    color: colors.ink, margin: 0, valign: 'mid', ...options });
}

function page(title) {
  const slide = pptx.addSlide();
  slide.background = { color: colors.paper };
  text(slide, title, 0.7, 0.46, 11.95, 0.67,
    { fontFace: 'Aptos Display', fontSize: 31, bold: true, color: colors.accent });
  slide.addShape(pptx.ShapeType.line, { x: 0.7, y: 1.34, w: 11.95, h: 0,
    line: { color: colors.accent, width: 1.35 } });
  text(slide, 'T085  /  Phase 2  /  ' + pptx.slides.length, 0.7, 7.12, 4, 0.17,
    { fontSize: 10, color: colors.foot });
  return slide;
}

function lines(slide, copy, { x = 0.75, y = 1.78, w = 11.8, step = 0.75, fontSize = 23 } = {}) {
  copy.forEach((line, i) => text(slide, line, x, y + step * i, w, step - 0.08, { fontSize }));
}

function table(slide, headers, rows, widths, y = 1.72, fontSize = 17) {
  const header = headers.map((value) => ({ text: value,
    options: { bold: true, fill: colors.accent, color: colors.white } }));
  const body = rows.map((row, number) => row.map((value) => ({ text: String(value),
    options: { fill: number % 2 ? colors.stripe : colors.white } })));
  slide.addTable([header, ...body], { x: 0.75, y, colW: widths, rowH: 0.53,
    fontSize, fontFace: 'Aptos', color: colors.ink, margin: 0.09, autoPage: false,
    border: { type: 'solid', color: 'D5E1E0', pt: 0.6 }, valign: 'mid' });
}

async function chartImage(slide, filename, x, y, w, h) {
  const source = path.join(study, filename);
  const png = await fs.readFile(source);
  const aspect = png.readUInt32BE(16) / png.readUInt32BE(20);
  const width = Math.min(w, h * aspect);
  const height = width / aspect;
  slide.addImage({ path: source, x: x + (w - width) / 2, y: y + (h - height) / 2,
    w: width, h: height });
}

const subset = (group, model) => group.find((row) => row.model === model);
const testRanking = ranking.filter((row) => row.split === 'test');
const testOperating = operating.filter((row) => row.split === 'test');
const testCapacity = capacity.filter((row) => row.split === 'test');

let slide = page('Card fraud triage');
text(slide, 'Gaussian mixture and denoising autoencoder', 0.75, 2.28, 11.75, 0.7,
  { fontSize: 35, bold: true });
text(slide, 'Team T085  /  Section E  /  Project 26', 0.75, 3.43, 11.55, 0.4,
  { fontSize: 22 });

slide = page('Fraud ranking under a review budget');
lines(slide, ['Which score puts more fraud near the front of the review queue?',
  'Compare both models on matching chronological splits.',
  'The historical test split had already been inspected in Phase 1.']);
slide.addNotes('Topic 26 asks for a chronological, legitimate-only anomaly detection comparison and an untouched mixed test set. The historical score evaluation uses the same inspected rows.');

slide = page('Chronological dataset partitions');
table(slide, ['Partition', 'Transactions', 'Frauds', 'Fraud rate'], [
  ['Train', '170,888', 360, '0.2107%'], ['Validation', '56,958', 57, '0.1001%'],
  ['Historical test', '56,961', 75, '0.1317%'],
], [3.15, 3, 2.55, 3.1]);
lines(slide, ['Both original pipelines start from 170,528 legitimate training rows.'], { y: 4.28 });
slide.addNotes('Data source: ULB Machine Learning Group, Credit Card Fraud Detection, https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud');

slide = page('Features and anomaly scores');
lines(slide, ['29 standardized inputs from V1–V28 and log(1 + Amount).',
  'Time orders the split. Class identifies legitimate rows and supports evaluation.',
  'GMM ranks by negative log density from 20 full-covariance components.',
  'The autoencoder ranks by mean squared reconstruction error.'], { step: 0.83 });

slide = page('Original autoencoder architecture');
text(slide, '29 inputs   ·   16 hidden   ·   8 latent   ·   16 hidden   ·   29 outputs',
  0.75, 1.82, 11.83, 0.61, { fontSize: 25, bold: true });
lines(slide, ['ReLU hidden units learn nonlinear structure. The linear output reconstructs signed inputs.',
  'Train against clean inputs with Gaussian noise of standard deviation 0.05.',
  'Adam uses batches of 512. The original run reached the 35-epoch cap.'],
  { y: 2.72, step: 0.88, fontSize: 20 });

slide = page('Checkpoint selection');
text(slide, 'Original autoencoder', 0.75, 1.82, 5.4, 0.42, { bold: true });
text(slide, 'Controlled ablations', 6.8, 1.82, 5.68, 0.42, { bold: true });
lines(slide, ['Random legitimate reserve', 'Fixed noisy input', 'Internal validation R²'],
  { x: 0.75, y: 2.46, w: 5.4, step: 0.7, fontSize: 20 });
lines(slide, ['Shared chronological legitimate reserve', 'Fresh corruption per epoch', 'Lowest clean reserve MSE'],
  { x: 6.8, y: 2.46, w: 5.7, step: 0.7, fontSize: 20 });
slide.addNotes('Scikit-learn documents the original internal early-stopping metric as validation R². https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html');

slide = page('Review policies');
lines(slide, ['Threshold policies cap false alerts on legitimate validation rows.',
  'Apply frozen cutoffs to later rows and measure actual FPR, recall and precision.',
  'Capacity policies take the top-scoring transactions from a batch.',
  'Use the same number of reviews for each model.'], { step: 0.84 });

slide = page('Ranking metrics on the historical test');
table(slide, ['Model', 'AP', 'PR AUC', 'ROC AUC', 'Fraud prevalence'],
  testRanking.map((row) => [row.model, decimal(row.average_precision), decimal(row.pr_auc_trapezoidal),
    decimal(row.roc_auc), percent(row.base_rate, 3)]), [3.05, 1.75, 1.95, 2.15, 2.9]);
lines(slide, ['The GMM places fraud earlier in the queue.',
  'The autoencoder has higher ROC AUC and substantially lower AP.'], { y: 3.8 });

slide = page('Precision–recall curves');
text(slide, 'Frozen scores on validation and on the previously inspected test split.',
  0.75, 1.62, 11.65, 0.39, { fontSize: 18 });
await chartImage(slide, 'precision_recall.png', 0.85, 2.18, 11.55, 4.57);

slide = page('Historical recall and false alerts by budget');
const names = ['GMM', 'Autoencoder'];
const budgetRows = ['0.001', '0.005', '0.01', '0.05'].map((budget) => [
  percent(budget), ...names.flatMap((model) => {
    const row = subset(testOperating.filter((entry) => entry.fpr_budget === budget), model);
    return [percent(row.recall), row.fp, percent(row.fpr, 3)];
  }),
]);
table(slide, ['Validation FPR', 'GMM recall', 'GMM FP', 'GMM FPR', 'AE recall', 'AE FP', 'AE FPR'],
  budgetRows, [2.12, 1.63, 1.51, 1.63, 1.65, 1.49, 1.77], 1.72, 15);
lines(slide, ['At 5%, both catch 60 frauds and each sends more than 2,000 false alerts.'],
  { y: 4.81, fontSize: 19 });

slide = page('Fraud caught at equal workload');
const workloadRows = ['56', '100', '284', '569'].map((k) => [
  k, ...names.flatMap((model) => {
    const row = subset(testCapacity.filter((entry) => entry.k === k), model);
    return [row.tp, percent(row.precision_at_k, 2)];
  }),
]);
table(slide, ['Reviews', 'GMM frauds', 'GMM precision', 'AE frauds', 'AE precision'],
  workloadRows, [2.15, 2.2, 2.65, 2.15, 2.65]);
lines(slide, ['With 284 reviews, GMM catches 57 of 75 frauds. The autoencoder catches 33.'],
  { y: 4.81, fontSize: 19 });

slide = page('Ablations on mixed validation');
table(slide, ['Bottleneck', 'Input noise SD', 'Mean AP', 'Across-seed SD', 'Mean recall'],
  ablation.map((row) => [row.width, Number(row.noise_std).toFixed(2), decimal(row.mean_ap),
    decimal(row.sd_ap), percent(row.mean_recall)]), [2.15, 2.6, 2.1, 2.55, 2.4], 1.72, 16);
lines(slide, ['Three seeds per row. Recall is measured at a 0.5% validation FPR budget.',
  'Width 16 leads the tested autoencoders. Frozen GMM validation AP is 0.6299.'],
  { y: 4.78, step: 0.56, fontSize: 18 });

slide = page('Reconstruction learning curves');
await chartImage(slide, 'training_curves.png', 0.7, 1.62, 8.2, 5.18);
lines(slide, ['Clean reserve MSE selects each checkpoint.',
  'The dashed line shows the training objective for seed 26.',
  'Each model has up to 35 epochs.'],
  { x: 9.12, y: 2.05, w: 3.4, step: 1.18, fontSize: 17 });

slide = page('Hybrid score fusion');
text(slide, 'GMM density and autoencoder reconstruction', 0.75, 1.76, 11.8, 0.55,
  { fontSize: 27, bold: true });
lines(slide, ['Map both raw scores to rarity using legitimate validation transactions.',
  'Compare five autoencoder weights using mixed-validation AP.',
  'Freeze the selected weights and calibrate thresholds before evaluating the historical test.'],
  { y: 2.55, step: 0.78, fontSize: 20 });
text(slide, `Selected mixture  ${percent(hybridMeta.gmm_weight, 0)} GMM and ${percent(hybridMeta.ae_weight, 0)} autoencoder`,
  0.75, 5.35, 11.8, 0.55, { bold: true, color: colors.accent });

slide = page('Hybrid and standalone results');
const hybridRows = ['GMM', 'Autoencoder', 'Hybrid'].map((model) => {
  const validation = subset(hybridRanking.filter((row) => row.split === 'validation'), model);
  const test = subset(hybridRanking.filter((row) => row.split === 'test'), model);
  const queue = subset(hybridCapacity.filter((row) => row.split === 'test' && row.k === '284'), model);
  return [model, decimal(validation.average_precision), decimal(test.average_precision), queue.tp];
});
table(slide, ['Model', 'Validation AP', 'Historical test AP', 'Frauds at 284 reviews'], hybridRows,
  [2.95, 2.45, 2.95, 3.45]);
const onePercent = hybridOperating.filter((row) => row.split === 'test' && row.fpr_budget === '0.01');
const hybridPoint = subset(onePercent, 'Hybrid');
const gmmPoint = subset(onePercent, 'GMM');
lines(slide, [`At the 1% validation FPR budget, both catch ${hybridPoint.tp} frauds.`,
  `Hybrid produces ${hybridPoint.fp} false alerts. GMM produces ${gmmPoint.fp}.`,
  'The historical comparison shows similar workload and lower hybrid AP.'],
  { y: 4.23, step: 0.53, fontSize: 18 });

slide = page('Findings and holdout status');
lines(slide, ['GMM led the historical comparison by AP. The hybrid matched its fixed-capacity counts.',
  'Width 16 led the tested neural variants on mixed validation.',
  'An unseen batch with compatible features is still needed to confirm the results.'],
  { step: 0.94, fontSize: 21 });

const packageZip = await JSZip.loadAsync(await pptx.write({ outputType: 'nodebuffer' }));
const manifestName = '[Content_Types].xml';
const manifest = await packageZip.file(manifestName).async('string');
// Some PptxGenJS releases declare a master for each slide while emitting a single master.
const masterOverride = /<Override\s+PartName="\/(ppt\/slideMasters\/slideMaster\d+\.xml)"[^>]*\/>/g;
packageZip.file(manifestName, manifest.replace(masterOverride, (entry, part) => packageZip.file(part) ? entry : ''));
await fs.writeFile(destination, await packageZip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
