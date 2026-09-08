/** OpenDetect multi-worker relay. Deploy execute-as OWNER A, never worker.
 * Workers receive per-file resumable upload capabilities, NEVER A's OAuth token.
 * Sticky leases require explicit release after a lost VM; no silent takeover.
 * No source/checkpoint deletion endpoint. Immutable snapshots accrue in A.
 */
const PROJECT = '1Sry3j9KonUEQ88besc12_zxkyI0DoXfI';
const OUTPUTS = '1o8-O578YTfc9FAwGKBZvDGJlzUYE7awI';
const SCENARIOS = ['A-1','A-2','A-3','B-1','B-2','B-3','C-1','C-2'];
const DATASETS = ['USTC_1c_train.npz','USTC_1c_test.npz','mal_32_1c_train.npz','mal_32_1c_test.npz','combined_train_data.npz','combined_test_data.npz'];
const FIELDS = 'id,name,size,md5Checksum,sha256Checksum,mimeType,parents,owners(emailAddress),appProperties,trashed';
function props() { return PropertiesService.getScriptProperties(); }
function sha(s) { return Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256,s,Utilities.Charset.UTF_8).map(x=>('0'+(x&255).toString(16)).slice(-2)).join(''); }
function check(v,m) { if (!v) throw new Error(m); }
function json(v) { return ContentService.createTextOutput(JSON.stringify(v)).setMimeType(ContentService.MimeType.JSON); }
function request(url,options) {
  options=options||{}; options.headers=Object.assign({},options.headers,{Authorization:'Bearer '+ScriptApp.getOAuthToken()});
  options.muteHttpExceptions=true;
  const r=UrlFetchApp.fetch(url,options);
  check(r.getResponseCode()>=200 && r.getResponseCode()<300,'Drive request failed ('+r.getResponseCode()+')');
  return r;
}
function api(path,method,body) {
  const o={method:method||'get'};
  if (body!==undefined) { o.contentType='application/json'; o.payload=JSON.stringify(body); }
  return JSON.parse(request('https://www.googleapis.com/drive/v3/'+path,o).getContentText()||'{}');
}
function metadata(id) { check(/^[\w-]+$/.test(id),'Invalid ID'); return api('files/'+id+'?fields='+encodeURIComponent(FIELDS)); }
function owner(r) { check(r.owners && r.owners.length===1,'My Drive owner required');return r.owners[0].emailAddress.toLowerCase(); }
function ownerA() {
  const a=owner(metadata(PROJECT));
  const me=api('about?fields=user(emailAddress),storageQuota');
  check(me.user.emailAddress.toLowerCase()===a,'Deploy execute-as origin A');
  return a;
}
function list(folder) {
  let result=[],token='';
  do {
    const r=api('files?q='+encodeURIComponent("'"+folder+"' in parents and trashed=false")+'&fields='+encodeURIComponent('nextPageToken,files('+FIELDS+')')+'&pageSize=1000'+(token?'&pageToken='+encodeURIComponent(token):''));
    result=result.concat(r.files||[]);token=r.nextPageToken||'';
    check(result.length<=2000,'Folder too large; owner review required');
  } while(token);
  return result;
}
function named(folder,name,create) {
  const found=list(folder).filter(r=>r.name===name);
  check(found.length<=1,'Duplicate folder/name: '+name);
  if(found.length) {check(found[0].mimeType==='application/vnd.google-apps.folder','Expected folder');return found[0].id;}
  check(create,'Required folder missing: '+name);
  return api('files?fields=id','post',{name:name,mimeType:'application/vnd.google-apps.folder',parents:[folder]}).id;
}
function validPath(path) {
  check(typeof path==='string' && path.length<=240 && !path.includes('..') && !path.includes('\\') && !path.startsWith('/'),'Invalid path');
  check(/^[A-Za-z0-9_./-]+$/.test(path) && !path.split('/').some(x=>!x || x.startsWith('.')),'Invalid path segments');
  check(!path.endsWith('.writing') && !path.startsWith('_relay/'),'Uncommitted or relay path');
  return path;
}
function state(s) { return JSON.parse(props().getProperty('STATE_'+s)||'null'); }
function putState(s,v) { props().setProperty('STATE_'+s,JSON.stringify(v)); }
function manifest(st) { return st.current ? JSON.parse(request('https://www.googleapis.com/drive/v3/files/'+st.current+'?alt=media').getContentText()) : {files:[],generation:0}; }
function saveJson(folder,name,data) { return DriveApp.getFolderById(folder).createFile(name,JSON.stringify(data),MimeType.PLAIN_TEXT).getId(); }
function auth(q) {
  check(typeof q.key==='string' && q.key.length>=32,'Worker secret required');
  const grant=JSON.parse(props().getProperty('KEY_'+sha(q.key))||'null');
  check(grant && !grant.revoked && grant.scenarios.includes(q.scenario),'Worker not assigned to scenario');
  check(SCENARIOS.includes(q.scenario),'Invalid scenario');
  return grant;
}
function lease(st,q,g) { check(st.lease && st.lease.id===q.lease && st.lease.worker===g.worker,'Lease lost; stop training'); }
function scan(folder,prefix,out,a) {
  list(folder).forEach(r=>{
    if(r.name==='_relay' || r.name.endsWith('.writing') || r.name.startsWith('STAGED_') || r.name.startsWith('BEFORE_')) return;
    const path=prefix+r.name;validPath(path);
    if(r.mimeType==='application/vnd.google-apps.folder') scan(r.id,path+'/',out,a);
    else {check(owner(r)===a && r.sha256Checksum,'Bootstrap needs A-owned SHA256-verifiable files');out.push({path:path,id:r.id,size:Number(r.size),sha256:r.sha256Checksum});}
    check(out.length<=300,'Too many output files');
  });
}
/** Owner-only editor action, not exposed through doPost.
 * Set script properties ACK_PRODUCERS_STOPPED=yes before bootstrap.
 */
function initializeOwner() {
  const a=ownerA();check(props().getProperty('ACK_PRODUCERS_STOPPED')==='yes','Stop old training A/B first');
  check(owner(metadata(OUTPUTS))===a,'Outputs must belong to A');
  props().setProperty('OWNER',a);
  SCENARIOS.forEach(s=>{
    if(state(s)) return;
    const out=named(OUTPUTS,s,true),relay=named(out,'_relay',true),objects=named(relay,'objects',true),snapshots=named(relay,'snapshots',true);
    const files=[];scan(out,'',files,a);
    check(new Set(files.map(f=>f.path)).size===files.length,'Duplicate bootstrap paths; owner review required');
    const current=saveJson(snapshots,'bootstrap.json',{generation:0,scenario:s,files:files});
    putState(s,{out:out,relay:relay,objects:objects,snapshots:snapshots,current:current,generation:0,lease:null});
  });
  const d=named(named(PROJECT,'data',false),'dataset',false), files=list(d);
  const map={};DATASETS.forEach(n=>{const f=files.filter(x=>x.name===n);check(f.length===1 && owner(f[0])===a && /^[a-f0-9]{64}$/.test(f[0].sha256Checksum),'Dataset ambiguous/wrong owner/missing hash');map[n]={id:f[0].id,size:Number(f[0].size),sha256:f[0].sha256Checksum};});
  props().setProperty('DATA',JSON.stringify(map));
  console.log('Initialized. Existing canonical data retained; new snapshots are under outputs/<scenario>/_relay.');
}
/** Set PENDING_WORKER to a label and PENDING_SCENARIOS to e.g. C-1,B-2.
 * Secret printed ONLY in private Apps Script owner execution log, not notebook.
 */
function issueWorker() {
  ownerA();const worker=props().getProperty('PENDING_WORKER'),scenarios=(props().getProperty('PENDING_SCENARIOS')||'').split(',').map(x=>x.trim());
  check(worker && /^[A-Za-z0-9_-]{1,50}$/.test(worker) && scenarios.every(x=>SCENARIOS.includes(x)),'Set pending worker/scenarios');
  const key=Utilities.getUuid().replace(/-/g,'')+Utilities.getUuid().replace(/-/g,'');
  props().setProperty('KEY_'+sha(key),JSON.stringify({worker:worker,scenarios:scenarios}));
  console.log('Give ONLY this worker its scoped Colab Secret OPENDETECT_WORKER_KEY: '+key);
}
/** Owner-only revoke all keys for PENDING_WORKER; subsequent requests fail. */
function revokeWorker() {
  ownerA();const worker=props().getProperty('PENDING_WORKER');check(worker,'Select worker');
  const lock=LockService.getScriptLock();lock.waitLock(30000);
  try {
    const all=props().getProperties();
    Object.keys(all).filter(k=>k.startsWith('KEY_')).forEach(k=>{
      const g=JSON.parse(all[k]);if(g.worker===worker) {g.revoked=true;props().setProperty(k,JSON.stringify(g));}
    });
  } finally {lock.releaseLock();}
}
/** Owner-only emergency fencing: stop worker FIRST; set PENDING_SCENARIOS. */
function releaseLostWorker() {
  ownerA();check(props().getProperty('ACK_PRODUCERS_STOPPED')==='yes','Confirm stopped workers');
  const lock=LockService.getScriptLock();lock.waitLock(30000);
  try {(props().getProperty('PENDING_SCENARIOS')||'').split(',').forEach(s=>{s=s.trim();check(SCENARIOS.includes(s),'Invalid scenario');const st=state(s);st.lease=null;putState(s,st);});} finally {lock.releaseLock();}
}
function doGet() { return json({service:'OpenDetect owner relay v1',ready:!!props().getProperty('DATA')}); }
function doPost(e) {
  try {
    check(e.postData.contents.length<180000,'Control request too large');
    const q=JSON.parse(e.postData.contents),g=auth(q);
    if(['hello','data','read'].includes(q.action)) return json({ok:true,result:dispatch(q,g)});
    const lock=LockService.getScriptLock();
    lock.waitLock(30000);
    try {return json({ok:true,result:dispatch(q,g)});} finally {lock.releaseLock();}
  } catch(err) {return json({ok:false,error:String(err.message).slice(0,300)});}
}
function dispatch(q,g) {
  check(Session.getEffectiveUser().getEmail().toLowerCase()===props().getProperty('OWNER'),'Service must execute as A');
  const st=state(q.scenario);check(st && props().getProperty('DATA'),'Owner setup incomplete');
  if(q.action==='hello') return {worker:g.worker,scenario:q.scenario,generation:st.generation,layout:'immutable-snapshot-v1'};
  if(q.action==='data') return JSON.parse(props().getProperty('DATA'));
  if(q.action==='claim') {
    check(typeof q.session==='string' && /^[a-f0-9]{32}$/.test(q.session),'Session required');
    check(!st.lease || (st.lease.id===q.session && st.lease.worker===g.worker),'Scenario occupied; no automatic takeover');
    st.lease={id:q.session,worker:g.worker};putState(q.scenario,st);
    return {lease:q.session,manifest:manifest(st)};
  }
  if(q.action==='read') {
    let r;
    if(q.dataset) r=JSON.parse(props().getProperty('DATA'))[q.dataset];
    else {lease(st,q,g);r=manifest(st).files.find(f=>f.path===validPath(q.path));}
    check(r,'File not authorized');
    check(Number.isSafeInteger(q.offset) && q.offset>=0 && q.offset<r.size,'Invalid offset');
    const end=Math.min(q.offset+8*1024*1024,r.size)-1;
    const response=request('https://www.googleapis.com/drive/v3/files/'+r.id+'?alt=media',{headers:{Range:'bytes='+q.offset+'-'+end}});
    const bytes=response.getContent();check(bytes.length===end-q.offset+1,'Range mismatch');
    return {data:Utilities.base64Encode(bytes),end:end+1,size:r.size,sha256:r.sha256};
  }
  lease(st,q,g);
  if(q.action==='release') {st.lease=null;putState(q.scenario,st);return {released:true};}
  if(q.action==='begin') {
    validPath(q.path);check(Number.isSafeInteger(q.size) && q.size>0 && q.size<=2*1024*1024*1024,'File size invalid');
    check(/^[a-f0-9]{64}$/.test(q.sha256),'SHA256 required');
    const quota=api('about?fields=storageQuota').storageQuota;
    if(quota.limit) check(Number(quota.limit)-Number(quota.usage)>q.size+1024*1024*1024,'A storage low; stop rather than drop checkpoint');
    const id=api('files/generateIds?count=1&space=drive&type=files').ids[0];
    const body={id:id,name:q.sha256+'__'+q.path.split('/').pop(),parents:[st.objects],
      appProperties:{scenario:q.scenario,path:q.path,sha256:q.sha256,lease:q.lease}};
    const response=request('https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable',{method:'post',contentType:'application/json',payload:JSON.stringify(body),headers:{'X-Upload-Content-Type':'application/octet-stream','X-Upload-Content-Length':String(q.size)}});
    const headers=response.getAllHeaders(),url=headers.Location||headers.location;
    check(url && url.startsWith('https://www.googleapis.com/'),'Unexpected upload host');
    return {id:id,url:url};
  }
  if(q.action==='commit') {
    check(q.generation===st.generation,'Stale generation');
    check(Array.isArray(q.files) && q.files.length<=300,'Invalid snapshot');
    const previous=manifest(st), old={};previous.files.forEach(f=>{old[f.path]=f;});
    const seen={};let total=0;
    q.files.forEach(f=>{
      validPath(f.path);check(!seen[f.path],'Duplicate path');seen[f.path]=true;total+=f.size;
      if(old[f.path] && ['path','id','size','sha256'].every(k=>old[f.path][k]===f[k])) return;
      const m=metadata(f.id);
      check(!m.trashed && owner(m)===props().getProperty('OWNER') && m.parents.includes(st.objects) && m.appProperties.scenario===q.scenario && m.appProperties.path===f.path && m.appProperties.lease===q.lease,'Wrong staged object');
      check(Number(m.size)===f.size && m.sha256Checksum===f.sha256,'Cloud hash/size mismatch');
    });
    check(total<=30*1024*1024*1024,'Snapshot too large');
    // Never remove paths already committed; no deletion semantics for workers.
    previous.files.forEach(f=>check(seen[f.path],'Snapshot omits previous path'));
    const config=q.files.find(f=>f.path==='GROUPED_CONFIG.json');check(config,'Config required');
    if(old['GROUPED_CONFIG.json']) check(config.sha256===old['GROUPED_CONFIG.json'].sha256,'Scientific config changed');
    const cfg=JSON.parse(request('https://www.googleapis.com/drive/v3/files/'+config.id+'?alt=media').getContentText());
    check(cfg.scenario===q.scenario,'Config scenario mismatch');
    const next={generation:st.generation+1,scenario:q.scenario,worker:g.worker,files:q.files};
    const id=saveJson(st.snapshots,'generation_'+next.generation+'.json',next);
    // Verify persistence before the single pointer update. Old snapshot stays valid.
    check(JSON.stringify(JSON.parse(request('https://www.googleapis.com/drive/v3/files/'+id+'?alt=media').getContentText()))===JSON.stringify(next),'Snapshot readback failed');
    st.previous=st.current;st.current=id;st.generation=next.generation;putState(q.scenario,st);
    return {generation:st.generation,snapshot:id,owner:'A'};
  }
  throw new Error('Unsupported action');
}
