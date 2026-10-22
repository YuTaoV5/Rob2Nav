import {useEffect,useRef,useState} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {ArrowLeft,ArrowRight,ArrowUpRight,BookOpen,Check,ChevronRight,Copy,FileText,Search,X} from 'lucide-react';
import {recommendedReports,reportCategories,reportCategory,reportDisplayTitle,reportKind,reportLink,sortReports,type Report} from './reportOrganization';
const query=()=>new URLSearchParams(location.hash.split('?')[1]||'');
const file=(path:string)=>'/api/files/'+path.split('/').map(encodeURIComponent).join('/');

function Reader({current,reports,category}:{current:Report;reports:Report[];category:string}){
 const [text,setText]=useState(''),[error,setError]=useState(''),[loading,setLoading]=useState(true),[copied,setCopied]=useState(false),[outline,setOutline]=useState<{id:string;text:string;level:number}[]>([]);
 const article=useRef<HTMLElement>(null),copyTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined);
 const group=reportCategories.find(c=>c.id===category)!;
 useEffect(()=>{const controller=new AbortController();setLoading(true);setError('');setText('');fetch('/api/reports/'+current.id,{signal:controller.signal}).then(async r=>{if(!r.ok)throw Error('文档加载失败，请稍后重试。');return r.json()}).then(d=>setText(d.text)).catch(e=>{if(e.name!=='AbortError')setError(e.message)}).finally(()=>{if(!controller.signal.aborted)setLoading(false)});return()=>controller.abort()},[current.id]);
 useEffect(()=>{if(loading||!article.current)return;const headings=[...article.current.querySelectorAll('h2,h3')].filter(h=>h.getClientRects().length);setOutline(headings.map((h,i)=>{h.id='report-section-'+i;return {id:h.id,text:h.textContent||'',level:Number(h.tagName.slice(1))}}))},[text,loading]);
 useEffect(()=>()=>{if(copyTimer.current)clearTimeout(copyTimer.current)},[]);
 const resolvePath=(src:string)=>{try{src=decodeURIComponent(src)}catch{}const base=current.path.split('/').slice(0,-1).join('/');const parts=(src.startsWith('/')?src:base+'/'+src).split('/');const out:string[]=[];for(const p of parts){if(p==='..')out.pop();else if(p&&p!=='.')out.push(p)}return out.join('/')};
 const resolve=(src:string)=>/^(https?:|data:)/.test(src)?src:file(resolvePath(src));
 const related=sortReports(reports.filter(r=>r.id!==current.id&&reportCategory(r)===category)).slice(0,4);
 async function copy(){try{await navigator.clipboard.writeText(text);setCopied(true);copyTimer.current=setTimeout(()=>setCopied(false),2000)}catch{setError('复制失败，可使用下方“原始文档”打开并复制。')}}
 return <div className="report-reader">
  <div className="report-reader-bar"><a href={'#reports?category='+category}><ArrowLeft size={14}/>返回{group.name}</a><div><a href={file(current.path)} target="_blank" rel="noreferrer">原始文档<ArrowUpRight size={13}/></a><button onClick={copy} disabled={loading||!text}>{copied?<Check size={13}/>:<Copy size={13}/>} {copied?'已复制':'复制原文'}</button></div></div>
  <nav className="report-breadcrumb" aria-label="文档位置"><a href="#reports">研究报告</a><ChevronRight size={12}/><a href={'#reports?category='+category}>{group.name}</a><ChevronRight size={12}/><span>{reportKind(current)}</span></nav>
  <header className="report-reader-heading"><h1>{reportDisplayTitle(current)}</h1><details><summary>文档来源与文件路径</summary><span>{current.path}</span>{current.name!==reportDisplayTitle(current)&&<p>原文标题：{current.name}</p>}</details></header>
  <div className="report-reading-layout"><div className="report-paper">{error&&<div className="report-error" role="alert">{error}</div>}{loading?<p className="report-loading">正在读取文档…</p>:<article className="markdown report-markdown" ref={article}><ReactMarkdown remarkPlugins={[remarkGfm]} components={{
   h1:({children})=><h2 className="report-original-heading">{children}</h2>,
   img:({src,alt})=><img src={resolve(String(src||''))} alt={alt||''} loading="lazy"/>,
   a:({href,children})=>{const value=href||'';const path=resolvePath(value.split('#')[0]);const target=reports.find(r=>[r.path,...(r.aliases||[])].includes(path));if(value.startsWith('#'))return <a href={value} onClick={e=>{e.preventDefault();let slug=value.slice(1);try{slug=decodeURIComponent(slug)}catch{}const normalized=(s:string)=>s.trim().toLowerCase().replace(/[^\p{L}\p{N}]+/gu,'-').replace(/^-|-$/g,'');const found=[...(article.current?.querySelectorAll('h2,h3,h4')||[])].find(h=>normalized(h.textContent||'')===normalized(slug));found?.scrollIntoView({block:'start',behavior:'smooth'})}}>{children}</a>;return target?<a href={reportLink(target)}>{children}</a>:<a href={/^(https?:)/.test(value)?value:resolve(value)} target="_blank" rel="noreferrer">{children}</a>}
  }}>{text}</ReactMarkdown></article>}</div><aside className="report-reading-aside">{outline.length>0&&<><span className="report-kicker">本文目录</span><nav aria-label="本文目录">{outline.map(h=><button key={h.id} className={h.level===3?'subsection':''} onClick={()=>document.getElementById(h.id)?.scrollIntoView({block:'start',behavior:'smooth'})}>{h.text}</button>)}</nav></>}<div className="report-related"><span className="report-kicker">同主题阅读</span>{related.map(r=><a key={r.id} href={reportLink(r)}>{reportDisplayTitle(r)}<ArrowUpRight size={12}/></a>)}</div></aside></div>
 </div>;
}

export default function ReportsLibrary({reports:source}:{reports:Record<string,any>[]}){
 const reports=source as Report[];
 const [params,setParams]=useState(query),[search,setSearch]=useState('');
 useEffect(()=>{const change=()=>{setParams(query());window.scrollTo(0,0)};window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change)},[]);
 const path=params.get('path'),current=reports.find(r=>r.path===path||r.aliases?.includes(path||''));
 const selected=reportCategories.find(c=>c.id===params.get('category'));
 const categories=reportCategories.map(c=>({...c,items:sortReports(reports.filter(r=>reportCategory(r)===c.id))}));
 const needle=search.trim().toLowerCase();
 const matches=sortReports(reports.filter(r=>(!selected||reportCategory(r)===selected.id)&&(!needle||[reportDisplayTitle(r),r.name,r.path,...(r.aliases||[])].join(' ').toLowerCase().includes(needle))));
 const featured=recommendedReports.map(item=>({...item,report:reports.find(r=>r.path===item.path)})).filter(item=>item.report);
 return <div className="reports-v2">{current?<Reader key={current.id} current={current} reports={reports} category={reportCategory(current)}/>:<>
  <header className="report-library-heading"><div><span className="report-kicker">THE RESEARCH LIBRARY</span><h1>文档与研究<span>.</span></h1><p>操作指南、实验方法与验证记录，按使用场景整理。</p></div><span className="report-total">{reports.length}<small>篇已收录文档</small></span></header>
  {path&&!current&&<p className="report-error" role="alert">暂未找到该文档，请按名称搜索或从分类中查找。</p>}
  <div className="report-discovery-bar"><div className="report-search"><Search size={17}/><input aria-label="搜索文档" placeholder="搜索主题、技能或文档名称" value={search} onChange={e=>setSearch(e.target.value)}/>{search&&<button aria-label="清除搜索" onClick={()=>setSearch('')}><X size={14}/></button>}<span>{selected?'当前分类':'全部文档'}</span></div>{selected&&<a href="#reports" className="report-back"><ArrowLeft size={13}/>全部主题</a>}</div>
  {!selected&&!needle&&<><section className="report-start"><div className="report-section-heading"><h2>先读这几篇</h2><span>从使用到进展，找到你的起点</span></div><div className="report-featured-grid">{featured.map((f,i)=><a className={'report-featured '+(i===0?'lead':'')} href={reportLink(f.report!)} key={f.path}><span className="report-feature-label">{f.label}</span><h3>{f.title}</h3><p>{f.description}</p><span className="report-feature-footer">{reportDisplayTitle(f.report!)}<ArrowUpRight size={17}/></span></a>)}</div></section>
   <section><div className="report-section-heading"><h2>按主题浏览</h2><span>操作指南与研究记录分开阅读</span></div><div className="report-topics">{categories.map(c=><section className="report-topic" key={c.id}><div className="report-topic-top"><span className="report-kicker">{c.eyebrow}</span><span>{c.items.length} 篇</span></div><h3><a href={'#reports?category='+c.id}>{c.name}<ArrowUpRight size={15}/></a></h3><p>{c.description}</p><div className="report-topic-links">{c.items.slice(0,3).map(r=><a key={r.id} href={reportLink(r)}>{reportDisplayTitle(r)}<ChevronRight size={12}/></a>)}</div><a className="report-topic-more" href={'#reports?category='+c.id}>查看全部<ArrowRight size={13}/></a></section>)}</div></section></>}
  {(selected||needle)&&<><nav className="report-filter-tabs" aria-label="文档分类"><a href="#reports" className={!selected?'active':''}>全部</a>{categories.map(c=><a key={c.id} href={'#reports?category='+c.id} className={selected?.id===c.id?'active':''}>{c.name}<small>{c.items.length}</small></a>)}</nav><div className="report-section-heading report-results-heading"><div><h2>{needle?'搜索结果':selected?.name}</h2><p>{needle?`与“${search.trim()}”相关的文档`:selected?.description}</p></div><span>{matches.length} 篇</span></div>
   {matches.length?<div className="report-results">{matches.map(r=><a className="report-result" href={reportLink(r)} key={r.id}><span className="report-type-symbol"><FileText size={17}/></span><div><span className="report-result-meta">{reportCategories.find(c=>c.id===reportCategory(r))?.name}<i>·</i>{reportKind(r)}</span><h3>{reportDisplayTitle(r)}</h3></div><ArrowUpRight size={16}/></a>)}</div>:<div className="report-no-results"><BookOpen size={28}/><h3>没有找到相关文档</h3><p>试试技能名，例如“后空翻”或“导航”。</p><button onClick={()=>setSearch('')}>清除搜索</button></div>}</>}
  <div className="report-library-note">文档保留原始内容与来源。历史实验结论请结合对应版本、验证范围及交接记录阅读。</div>
 </>}</div>;
}
