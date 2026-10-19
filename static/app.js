'use strict';
const $ = (id) => document.getElementById(id);
const names = {cut:'Cut',defence:'Defence',glance:'Glance',offdrive:'Off drive',ondrive:'On drive',pull:'Pull',sweep:'Sweep',unorthodox:'Unorthodox'};
let selectedFile = null, lastResult = null, busy = false;
function setFile(file) {
  selectedFile = file;
  $('file-label').textContent = file ? file.name : 'Drop your video here';
  $('analyze-button').disabled = !file || busy;
}
$('video-input').addEventListener('change', e => setFile(e.target.files[0]));
$('dropzone').addEventListener('dragover', e => {e.preventDefault(); $('dropzone').classList.add('drag');});
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('drag'));
$('dropzone').addEventListener('drop', e => {e.preventDefault(); $('dropzone').classList.remove('drag'); setFile(e.dataTransfer.files[0]);});
function showState(state) {
  for (const name of ['empty-state','loading-state','error-state','prediction']) $(name).hidden = name !== state;
}
async function analyze(url, options) {
  if (busy) return;
  busy = true; $('analyze-button').disabled = true;
  document.querySelectorAll('.example').forEach(b => b.disabled = true);
  showState('loading-state');
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed. Please retry.');
    lastResult = data; renderResult(data); showState('prediction');
  } catch (error) {
    $('error-state').textContent = error.message; showState('error-state');
  } finally {
    busy = false; $('analyze-button').disabled = !selectedFile;
    document.querySelectorAll('.example').forEach(b => b.disabled = false);
  }
}
$('upload-form').addEventListener('submit', e => {
  e.preventDefault(); if (!selectedFile) return;
  const body = new FormData(); body.append('video', selectedFile);
  analyze('/api/predict', {method:'POST',body});
});
function renderResult(data) {
  $('decision').textContent = data.decision.toUpperCase();
  $('shot-label').textContent = data.display_label;
  $('confidence').textContent = `${(data.confidence*100).toFixed(1)}%`;
  $('prediction-context').textContent = data.example ? `Test example: ${names[data.example.label]}. ${data.example.note}` : 'Prediction for your uploaded clip';
  $('review-note').textContent = data.accepted ? 'Score exceeds the review threshold. Check the clip and the alternatives before relying on it.' : 'Score is below the review threshold. The label above is the top candidate and needs human review.';
  $('review-note').className = `review-note${data.accepted?'':' warn'}`;
  const scores = $('scores'); scores.replaceChildren();
  for (const [label,p] of Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1])) {
    const row=document.createElement('div'); row.className=`class-row${label===data.label?' top':''}`;
    const line=document.createElement('div'); line.className='class-label';
    const a=document.createElement('span'); a.textContent=names[label]||label;
    const b=document.createElement('span'); b.textContent=`${(p*100).toFixed(1)}%`;
    line.append(a,b);
    const track=document.createElement('div'); track.className='class-track';
    const fill=document.createElement('div'); fill.className='class-fill'; fill.style.width=`${p*100}%`;
    track.append(fill); row.append(line,track); scores.append(row);
  }
  $('timing').textContent = `${data.seconds.toFixed(2)} s on CPU`;
  const strip=$('phase-strip'); strip.replaceChildren();
  ['Early','Middle','Late'].forEach((name,phase) => {
    const group=document.createElement('div'); group.className='phase';
    const imgs=document.createElement('div'); imgs.className='phase-images';
    for(let i=0;i<3;i++) {const im=document.createElement('img'); im.src=data.previews[phase*3+i]; im.alt=`${name} sample ${i+1}`; imgs.append(im);}
    const caption=document.createElement('p'); caption.textContent=name; group.append(imgs,caption); strip.append(group);
  });
}
$('download-result').addEventListener('click', () => {
  if(!lastResult) return;
  const {previews,...exported}=lastResult;
  const url=URL.createObjectURL(new Blob([JSON.stringify(exported,null,2)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download='cricketshot-prediction.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function initialize() {
  try {
    const [examplesResponse,evidenceResponse]=await Promise.all([fetch('/api/examples'),fetch('/api/evidence')]);
    const examples=await examplesResponse.json();
    for(const e of examples) {
      const button=document.createElement('button'); button.className='example'; button.type='button';
      button.setAttribute('aria-label',`Analyze ${names[e.label]} test example`);
      const img=document.createElement('img'); img.src=`/api/example/${e.id}/thumbnail`; img.alt=`${names[e.label]} sample`;
      const caption=document.createElement('span'); caption.textContent=names[e.label]; button.append(img,caption);
      button.addEventListener('click',()=> {if(busy)return; document.querySelectorAll('.example').forEach(b=>b.classList.remove('selected')); button.classList.add('selected'); analyze(`/api/example/${e.id}`,{method:'POST'});});
      $('examples').append(button);
    }
    if(evidenceResponse.ok) {
      const evidence=await evidenceResponse.json(), m=evidence.models.phase_delta;
      const items=[[`${(m.test.accuracy*100).toFixed(1)}%`,'Test accuracy'],[`${(m.test.macro_f1*100).toFixed(1)}%`,'Macro F1'],[String(m.test.n),'Test clips']];
      for(const [value,label] of items) {const box=document.createElement('div');box.className='metric';const n=document.createElement('strong');n.textContent=value;const l=document.createElement('span');l.textContent=label;box.append(n,l);$('metrics').append(box);}
    }
  }catch(error){$('examples').textContent='Examples unavailable. Check the setup instructions.';}
}
initialize();
