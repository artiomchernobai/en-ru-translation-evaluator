export default function(component){
 const {parentElement,data}=component;
 const root=parentElement.querySelector('.review'),panes=root.querySelector('.panes'),status=root.querySelector('.selection');
 const nodes={source:new Map(),target:new Map()},texts={};
 const remembered=root._reviewIdentity===data.identity?root._selection:null;root._reviewIdentity=data.identity;root._selection=null;
 status.textContent='Нажмите на слово в любом тексте';
 const oldScroll={};panes.querySelectorAll('.text').forEach(el=>oldScroll[el.dataset.side]=el.scrollTop);
 panes.replaceChildren();
 const opposite={source:'target',target:'source'};
 function select(side,token){
  root._selection={side,id:token.id};
  root.querySelectorAll('.word[aria-pressed="true"]').forEach(w=>w.setAttribute('aria-pressed','false'));
  nodes[side].get(token.id).setAttribute('aria-pressed','true');
  root.querySelectorAll('.selected,.related').forEach(el=>{if(el.classList.contains('word'))el.classList.remove('selected','related');});
  nodes[side].get(token.id).classList.add('selected');
  const peers=token.partners.map(id=>nodes[opposite[side]].get(id)).filter(Boolean);
  peers.forEach(el=>el.classList.add('related'));
  if(peers.length){
   const pane=texts[opposite[side]],r=peers[0].getBoundingClientRect(),p=pane.getBoundingClientRect();
   if(r.top<p.top||r.bottom>p.bottom)pane.scrollTop+=r.top-p.top-p.height/2;
  }
  const descriptions={unlinked:'Связей не найдено',sentence_unaligned:'Группа предложений без пары',not_processed:'Блок не обработан моделью'};
  status.textContent=token.text+' · '+(peers.length?'Связей: '+peers.length:(descriptions[token.state]||'Связей не найдено'));
 }
 for(const side of ['source','target']){
  const payload=data.panes[side],pane=document.createElement('section'),header=document.createElement('div'),body=document.createElement('div');
  pane.className='pane';pane.setAttribute('aria-label',side==='source'?'Оригинал':'Перевод');
  header.className='pane-header';header.textContent=payload.document_id+' · '+payload.label;header.title=header.textContent;
  body.className='text';body.dataset.side=side;body.setAttribute('aria-label',side==='source'?'Текст оригинала':'Текст перевода');texts[side]=body;
  const chars=Array.from(payload.text);let offset=0;
  for(const token of payload.tokens){
   body.append(document.createTextNode(chars.slice(offset,token.start).join('')));
   const word=document.createElement('span');word.textContent=chars.slice(token.start,token.end).join('');
   word.className='word '+token.state+(token.block===data.focus_block?' current-block':'');word.dataset.id=token.id;word.dataset.block=token.block;
   word.setAttribute('role','button');word.tabIndex=0;word.setAttribute('aria-label',token.text);word.setAttribute('aria-pressed','false');
   word.onclick=()=>select(side,token);
   word.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();word.click();}};
   body.append(word);nodes[side].set(token.id,word);offset=token.end;
  }
  body.append(document.createTextNode(chars.slice(offset).join('')));
  if(!payload.text)body.textContent='В этой группе нет текста';
  pane.append(header,body);panes.append(pane);body.scrollTop=oldScroll[side]||0;
 }
 const size=()=>{
  const dock=document.querySelector('[data-testid="stBottom"]');
  const bottom=Math.min(window.innerHeight,dock?dock.getBoundingClientRect().top:window.innerHeight);
  const height=Math.max(120,Math.floor(bottom-root.getBoundingClientRect().top-12));
  if(root.style.height!==height+'px')root.style.height=height+'px';
 };
 size();window.addEventListener('resize',size);
 const dock=document.querySelector('[data-testid="stBottom"]');
 const observer=new ResizeObserver(size);if(dock)observer.observe(dock);const heading=document.querySelector('h1');if(heading)observer.observe(heading);
 if(data.focus_block!==null){
  for(const side of ['source','target']){
   const token=data.panes[side].tokens.find(t=>t.block===data.focus_block);
   if(token){const el=nodes[side].get(token.id),pane=texts[side];pane.scrollTop+=el.getBoundingClientRect().top-pane.getBoundingClientRect().top-20;}
  }
 }
 if(remembered){const token=data.panes[remembered.side].tokens.find(t=>t.id===remembered.id);if(token)select(remembered.side,token);}
 return ()=>{window.removeEventListener('resize',size);observer.disconnect();};
}
