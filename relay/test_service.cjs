// Pure authorization/lease tests; never touches Google or user Drive.
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const values={OWNER:'a@example.com',DATA:'{}'};
const p={getProperty:k=>values[k]||null,setProperty:(k,v)=>{values[k]=v;},getProperties:()=>({...values})};
const ctx={PropertiesService:{getScriptProperties:()=>p},Session:{getEffectiveUser:()=>({getEmail:()=> 'a@example.com'})},console};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(__dirname+'/Code.gs','utf8'),ctx);
ctx.sha=s=>require('node:crypto').createHash('sha256').update(s).digest('hex');
const secret='a'.repeat(64);
values['KEY_'+ctx.sha(secret)]=JSON.stringify({worker:'B-mn',scenarios:['C-1']});
assert.equal(ctx.auth({key:secret,scenario:'C-1'}).worker,'B-mn');
assert.throws(()=>ctx.auth({key:secret,scenario:'B-2'}));
assert.throws(()=>ctx.auth({key:'wrong'.repeat(12),scenario:'C-1'}));
values['KEY_'+ctx.sha(secret)]=JSON.stringify({worker:'B-mn',scenarios:['C-1'],revoked:true});
assert.throws(()=>ctx.auth({key:secret,scenario:'C-1'}));
for(const path of ['../x','/x','a\\b','a//b','.secret','x.writing','_relay/x']) assert.throws(()=>ctx.validPath(path));
assert.equal(ctx.validPath('resume_state/run/last.pt'),'resume_state/run/last.pt');
values['STATE_C-1']=JSON.stringify({generation:0,current:null,lease:null});
let q={scenario:'C-1',action:'claim',session:'a'.repeat(32)};
const g={worker:'B-mn'};
ctx.dispatch(q,g);
assert.throws(()=>ctx.dispatch({...q,session:'b'.repeat(32)},{worker:'B-mc'}));
assert.throws(()=>ctx.dispatch({scenario:'C-1',action:'release',lease:'b'.repeat(32)},g));
ctx.dispatch({scenario:'C-1',action:'release',lease:'a'.repeat(32)},g);
ctx.dispatch({...q,session:'b'.repeat(32)},{worker:'B-mc'});
assert.throws(()=>ctx.dispatch({scenario:'C-1',action:'release',lease:'a'.repeat(32)},g));
values['STATE_B-2']=JSON.stringify({generation:0,current:null,lease:null});
ctx.dispatch({scenario:'B-2',action:'claim',session:'c'.repeat(32)},g); // different scenario allowed
assert.throws(()=>ctx.lease({lease:{id:'x',worker:'other'}},{lease:'x'},g));
console.log('Service path, exclusivity, fencing and parallel-scenario tests passed');
