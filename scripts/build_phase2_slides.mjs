import pptxgen from '/Users/aad1tyayadav/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs/dist/pptxgen.cjs.js';
const pptx=new pptxgen();pptx.layout='LAYOUT_WIDE';pptx.author='Team T085';pptx.subject='Phase 2 card fraud triage';pptx.title='Card Fraud Triage: Phase 2';
const slides=[
['Card Fraud Triage | Phase 2','Team T085 - Section E\nDenoising autoencoder compared with the frozen GMM'],
['Question','Can nonlinear reconstruction rank fraud better than the Phase 1 density model?'],
['Same data, fair comparison','Original ULB/Kaggle dataset: 284,807 rows, 492 frauds\nChronological frozen split: 170,528 legitimate training rows\nValidation: 57 frauds; test: 75 frauds'],
['Input preparation','V1-V28 plus log(1+Amount): 29 features\nScaler fitted only on legitimate training rows\nTime orders the split; Class is excluded from model inputs'],
['Autoencoder design','29 → 16 → 8 → 16 → 29 neural network\nNoise standard deviation: 0.05; target: clean legitimate input\nAnomaly score: mean squared reconstruction error'],
['Training and selection','Adam, 512-row batches, seed 26\nUp to 35 epochs; 10% legitimate-only early-stop reserve\nMaximum epoch limit reached: convergence is not claimed'],
['Frozen GMM reference','20 full-covariance components\nAnomaly score: negative log density\nLoad Phase 1 scores on exactly the same validation and test rows'],
['Primary ranking result','Test average precision\nGMM: 0.5822\nAutoencoder: 0.0908\nTest fraud prevalence: 0.00132'],
['Precision-recall curve','Both models scored on the same 56,961 test transactions'],
['At the validation FPR budget','0.5% validation false-positive budget\nGMM: 57/75 frauds; 175 false alerts; 76% recall\nAutoencoder: 27/75 frauds; 135 false alerts; 36% recall'],
['At fixed review capacity','Top 0.5% = 284 test transactions reviewed\nGMM catches 57 frauds; precision 20.07%\nAutoencoder catches 33; precision 11.62%'],
['What the result means','The tested autoencoder ranks fraud less effectively than GMM\nROC AUC alone hides the rare-event review burden\nOne neural configuration does not settle all architectures'],
['Limitations','Only two historical days and 75 test frauds\nAnonymized features limit explanations\nNeural training reached its epoch cap\nTest partition had already been inspected in Phase 1'],
['Conclusion','Use AP, recall, precision and workload for fraud triage\nThe frozen GMM leads this tested comparison\nFurther neural tuning needs a predetermined validation-only protocol']
];
for(let i=0;i<slides.length;i++){let sl=pptx.addSlide();sl.background={color:'F7F8F5'};sl.addText(slides[i][0],{x:.65,y:.48,w:12,h:.72,fontFace:'Aptos Display',fontSize:30,bold:true,color:'147A89'});sl.addShape(pptx.ShapeType.line,{x:.68,y:1.35,w:11.95,h:0,line:{color:'147A89',width:2}});sl.addText(slides[i][1],{x:.8,y:1.75,w:11.7,h:4.8,fontFace:'Aptos',fontSize:i===0?27:24,color:'1B3542',breakLine:false,margin:0,paraSpaceAfterPt:17});if(i===8)sl.addImage({path:'results/phase2/test_pr_curve.png',x:6.2,y:2.15,w:6.1,h:4.2});sl.addText(`T085  |  Phase 2  |  ${i+1}`,{x:.65,y:7.1,w:4,h:.2,fontSize:10,color:'617A84'});}
await pptx.writeFile({fileName:'docs/phase2/T085_Phase2.pptx'});
