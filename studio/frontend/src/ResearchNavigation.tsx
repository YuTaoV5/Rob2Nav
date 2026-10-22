const sections=[['capabilities','能力总览','了解能力'],['training','训练方案','选择方法'],['compare','基准对比','验证结果'],['models','模型与证据','追溯资产'],['reports','研究报告','查阅文档']];
export default function ResearchNavigation({page}:{page:string}){
 return <nav className="research-navigation" aria-label="研究工作区"><span className="research-nav-label">研究工作区</span><div>{sections.map(([id,label,hint])=><a key={id} href={'#'+id} aria-current={page===id?'page':undefined}><strong>{label}</strong><small>{hint}</small></a>)}</div></nav>
}
