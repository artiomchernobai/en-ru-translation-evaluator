export default function(component){
 const {parentElement,data,setTriggerValue}=component;
 const root=parentElement.querySelector('.candidate-reader'),panes=root.querySelector('.candidate-panes');
 const focusKey=JSON.stringify([data.identity,data.selection,data.mode]);
 const changed=root._focusKey!==focusKey;
 const scroll={};panes.querySelectorAll('.candidate-text').forEach(el=>scroll[el.dataset.side]=el.scrollTop);
 root._focusKey=focusKey;panes.replaceChildren();
 const focus=[];
 for(const side of ['source','target']){
  const pane=document.createElement('section'),heading=document.createElement('div'),text=document.createElement('div');
  pane.className='candidate-pane';pane.setAttribute('aria-label',side==='source'?'Оригинал кандидатов':'Перевод кандидатов');
  heading.className='candidate-heading';heading.textContent=data.panes[side].document+(data.panes[side].missing_center?' · нет текста центральной группы':'');
  text.className='candidate-text';text.dataset.side=side;text.tabIndex=0;
  for(const record of data.panes[side].sentences){
   const sentence=document.createElement('div');sentence.className='candidate-sentence'+(record.context?' context':'');sentence.dataset.sentence=record.index;
   for(const part of record.pieces){
    const span=document.createElement('span');span.className='candidate-piece'+(part.suspect?' suspect':'')+(part.active?' active':'');span.textContent=part.text;
    if(part.pick!==null){
     span.setAttribute('role','button');span.tabIndex=0;span.dataset.candidate=part.pick;
     span.onclick=()=>setTriggerValue('pick',{identity:data.identity,id:part.pick});
     span.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();span.click();}};
    }
    sentence.append(span);
   }
   text.append(sentence);if(record.focus)focus.push([text,sentence]);
  }
  if(!text.childNodes.length)text.textContent='В этом фрагменте нет текста';
  pane.append(heading,text);panes.append(pane);text.scrollTop=scroll[side]||0;
 }
 function size(){
  const dock=document.querySelector('[data-testid="stBottom"]');
  const bottom=Math.min(window.innerHeight,dock?dock.getBoundingClientRect().top:window.innerHeight);
  root.style.height=Math.max(180,Math.floor(bottom-root.getBoundingClientRect().top-12))+'px';
 }
 function center(){if(changed)for(const [body,sentence] of focus){body.scrollTop+=sentence.getBoundingClientRect().top-body.getBoundingClientRect().top-body.clientHeight/2+sentence.offsetHeight/2;}}
 size();const frame=requestAnimationFrame(center);
 window.addEventListener('resize',size);
 const observer=new ResizeObserver(size),dock=document.querySelector('[data-testid="stBottom"]');if(dock)observer.observe(dock);
 return ()=>{cancelAnimationFrame(frame);window.removeEventListener('resize',size);observer.disconnect();};
}
