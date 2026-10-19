'use strict';
const $=id=>document.getElementById(id);
const labels={'drive':'Drive','legglance-flick':'Leg glance / flick','pullshot':'Pull','sweep':'Sweep'};
let file=null,busy=false,result=null;
function choose(f){file=f;$('file-label').textContent=f?f.name:'Drop a cricket image here';$('analyze-button').disabled=!f||busy;}
$('image-input').addEventListener('change',e=>choose(e.target.files[0]));
$('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('drag');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('drag'));
$('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('drag');choose(e.dataTransfer.files[0]);});
function state(active){for(const id of ['empty-state','loading-state','error-state','prediction'])$(id).hidden=id!==active;}
async function predict(url,options){
  if(busy)return;busy=true;$('analyze-button').disabled=true;document.querySelectorAll('.example').forEach(b=>b.disabled=true);state('loading-state');
  try{const response=await fetch(url,options),data=await response.json();if(!response.ok)throw new Error(data.error||'Recognition failed.');result=data;render(data);state('prediction');}
  catch(error){$('error-state').textContent=error.message;state('error-state');}
  finally{busy=false;$('analyze-button').disabled=!file;document.querySelectorAll('.example').forEach(b=>b.disabled=false);}
}
$('image-form').addEventListener('submit',e=>{e.preventDefault();if(!file)return;const body=new FormData();body.append('image',file);predict('/image-api/predict',{method:'POST',body});});
function render(data){
  $('decision').textContent=data.decision.toUpperCase();$('shot-label').textContent=data.display_label;$('confidence').textContent=`${(data.confidence*100).toFixed(1)}%`;
  $('prediction-context').textContent=data.example?`Dataset label: ${labels[data.example.label]}. ${data.example.note}`:'Prediction for your image';
  const reason=!data.prediction_consistent?'The separate heads do not all agree with this prediction. Human review is needed.':data.confidence<data.threshold?'The score is below the review threshold. Human review is needed.':'All heads agree and the score exceeds the review threshold. A confident answer can still be wrong.';
  $('review-note').textContent=reason;$('review-note').className=`review-note${data.accepted?'':' warn'}`;
  const scores=$('scores');scores.replaceChildren();
  for(const [label,p] of Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1])){const row=document.createElement('div');row.className=`class-row${label===data.label?' top':''}`;const line=document.createElement('div');line.className='class-label';const name=document.createElement('span');name.textContent=labels[label];const val=document.createElement('span');val.textContent=`${(p*100).toFixed(1)}%`;line.append(name,val);const track=document.createElement('div');track.className='class-track';const fill=document.createElement('div');fill.className='class-fill';fill.style.width=`${p*100}%`;track.append(fill);row.append(line,track);scores.append(row);}
  const h=data.hog_baseline; $('hog-result').textContent=h?`HOG baseline: ${labels[h.label]} (${(100*h.score).toFixed(1)}% uncalibrated score). This uses grayscale gradient histograms. Neither model score is a guarantee of correctness.`:'';
  $('timing').textContent=`${(data.seconds*1000).toFixed(0)} ms on CPU`;
  const views=$('image-views');views.replaceChildren();
  ['Global view','Central crop'].forEach((name,i)=>{const box=document.createElement('div');box.className='model-view';const image=document.createElement('img');image.src=data.previews[i];image.alt=name;const p=document.createElement('p');const strong=document.createElement('strong');strong.textContent=name;p.append(strong,document.createTextNode(i===0?data.global_label:data.focus_label));box.append(image,p);views.append(box);});
  $('view-agreement').textContent='The global and crop heads are separate diagnostic classifiers. Their agreement controls the review decision. Displayed pixels match the model crops before normalization.';
}
$('download-result').addEventListener('click',()=>{if(!result)return;const {previews,...data}=result;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='cricketshot-image-prediction.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
async function init(){try{const [er,mr]=await Promise.all([fetch('/image-api/examples'),fetch('/image-api/evidence')]);const examples=await er.json();if(!examples.length)$('examples').textContent='Upload your own image, or restore selected examples using scripts/prepare_demo_examples.py (see README).';for(const e of examples){const b=document.createElement('button');b.className='example';b.type='button';b.setAttribute('aria-label',`Analyze ${labels[e.label]} example`);const im=document.createElement('img');im.src=`/image-api/example/${e.id}/thumbnail`;im.alt=labels[e.label];const t=document.createElement('span');t.textContent=e.short_label||labels[e.label];b.append(im,t);b.addEventListener('click',()=>{if(busy)return;document.querySelectorAll('.example').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');predict(`/image-api/example/${e.id}`,{method:'POST'});});$('examples').append(b);}if(mr.ok){const r=await mr.json(),m=r.models.contrast.test;for(const [value,label] of [[`${(m.accuracy*100).toFixed(1)}%`,'Test accuracy'],[`${(m.macro_f1*100).toFixed(1)}%`,'Macro F1'],[String(m.n),'Test images']]){const box=document.createElement('div');box.className='metric';const strong=document.createElement('strong');strong.textContent=value;const span=document.createElement('span');span.textContent=label;box.append(strong,span);$('metrics').append(box);}}const params=new URLSearchParams(location.search),example=params.get('example');if(params.get('focus')==='1')document.body.classList.add('focus-demo');if(examples.some(e=>e.id===example))await predict(`/image-api/example/${encodeURIComponent(example)}`,{method:'POST'});}catch(error){$('examples').textContent='Examples unavailable. See README.md for setup.';}}
init();
