"""
EVIRAG 48-Hour Comprehensive Multi-Domain Evaluation
=====================================================
5 domains × 50 queries × 5 systems = 1,250 evaluations

Domains
  D1  Homework & Academic Achievement   (education)
  D2  Statin Therapy & Cardiovascular   (biomedicine)
  D3  Minimum Wage & Labor Markets      (economics)
  D4  Climate Sensitivity & Attribution (earth sciences)
  D5  Dietary Fat & Cardiovascular      (nutrition science)

Systems (same ablation ladder as 12h experiment)
  SYS1  Vanilla RAG
  SYS2  Single-agent
  SYS3  EVIRAG-NoGraph
  SYS4  EVIRAG-NoMultiView
  SYS5  EVIRAG-Full

Runtime estimate: ~12-14 hours on RTX 4500 Ada (qwen3.6:35b-a3b)
Checkpoint: /home/snu/evirag_48h_checkpoint.json  (resume-safe)
"""

import json, re, time, os, math, random, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── configuration ─────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen3.6:35b-a3b"
SS_URL      = "https://api.semanticscholar.org/graph/v1/paper/search"

CHECKPOINT  = Path("/home/snu/evirag_48h_checkpoint.json")
RESULTS_OUT = Path("/home/snu/evirag_48h_results.json")
LOG_PATH    = Path("/home/snu/evirag_48h.log")

PAPERS_PER_DOMAIN  = 25
TOP_K_RETRIEVAL    = 12
COSINE_REL_THRESH  = 0.35
BOOTSTRAP_ITERS    = 1000
RANDOM_SEED        = 42
MAX_RUN_SECONDS    = 47 * 3600   # 47-hour hard limit

# ─── domain metadata ───────────────────────────────────────────────────────────
DOMAINS = {
    "homework": {
        "name": "Homework & Academic Achievement",
        "search_terms": [
            "homework academic achievement effects meta-analysis",
            "homework student learning outcomes research",
            "homework mental health wellbeing students",
            "homework grade level elementary high school effectiveness",
            "homework parental involvement academic performance",
        ],
        "controversy_class": "stable",
        "gt_conf_level": 0.5,
    },
    "statins": {
        "name": "Statin Therapy & Cardiovascular Outcomes",
        "search_terms": [
            "statin primary prevention cardiovascular randomized trial",
            "statin therapy side effects myopathy diabetes risk",
            "statin women cardiovascular benefit evidence",
            "statin all-cause mortality low-risk patients",
            "statin cognitive effects memory dementia",
        ],
        "controversy_class": "emerging",
        "gt_conf_level": 0.4,
    },
    "minwage": {
        "name": "Minimum Wage & Labor Markets",
        "search_terms": [
            "minimum wage employment effects causal identification",
            "minimum wage poverty reduction income inequality",
            "minimum wage youth employment effects evidence",
            "minimum wage monopsony labor market",
            "minimum wage small business price inflation",
        ],
        "controversy_class": "polarized",
        "gt_conf_level": 0.7,
    },
    "climate": {
        "name": "Climate Sensitivity & Attribution",
        "search_terms": [
            "equilibrium climate sensitivity CO2 doubling estimate",
            "attribution extreme weather events climate change",
            "aerosol forcing uncertainty climate projections",
            "climate model temperature sensitivity observation",
            "tipping points climate change irreversibility",
        ],
        "controversy_class": "stable",
        "gt_conf_level": 0.35,
    },
    "diet_cvd": {
        "name": "Dietary Fat & Cardiovascular Disease",
        "search_terms": [
            "saturated fat cardiovascular disease meta-analysis",
            "dietary fat heart disease randomized controlled trial",
            "Mediterranean diet cardiovascular mortality prevention",
            "LDL cholesterol dietary intervention cardiovascular risk",
            "dietary cholesterol heart disease eggs fat",
        ],
        "controversy_class": "polarized",
        "gt_conf_level": 0.65,
    },
}

# ─── queries (50 per domain) ───────────────────────────────────────────────────
QUERIES = {
  "homework": [
    {"q":"Does homework improve academic achievement?",
     "gt_views":["conditional positive effects grades 7-12 math","mixed or null meta-analytic evidence publication bias","excess homework wellbeing harm without achievement gain"],
     "gt_contradictions":[("positive achievement effect","null negative meta-analytic"),("wellbeing benefit","wellbeing harm anxiety")],"gt_conf":"medium"},
    {"q":"Does homework have differential effects across grade levels?",
     "gt_views":["substantially stronger effects secondary grades 7-12","negligible effects elementary counterproductive grade 3","grade effect artifact study design differences"],
     "gt_contradictions":[("stronger secondary effects","developmental counterproductivity"),("grade artifact","real grade effect")],"gt_conf":"medium"},
    {"q":"What is the effect of homework on student mental health?",
     "gt_views":["moderate homework no significant mental health harm","homework stress anxiety reduced sleep high-achieving schools","mental health effect mediated by autonomy intrinsic motivation"],
     "gt_contradictions":[("no harm moderate load","anxiety stress homework"),("autonomy mediates","direct harm hypothesis")],"gt_conf":"medium"},
    {"q":"Does parental involvement in homework improve learning outcomes?",
     "gt_views":["parental involvement positive academic achievement elementary","parental homework help anxiety reduce autonomy backfire","emotional support beneficial directive intervention counterproductive"],
     "gt_contradictions":[("positive parental involvement","backfire anxiety reduce"),("support vs directive","simple positive")],"gt_conf":"medium"},
    {"q":"Does homework develop non-cognitive skills such as self-regulation?",
     "gt_views":["homework develops study habits time management self-regulation","little rigorous evidence causal non-cognitive skill development","self-regulation prerequisite homework benefit not outcome"],
     "gt_contradictions":[("homework builds self-regulation","no causal evidence"),("prerequisite vs outcome","assumed direction")],"gt_conf":"low"},
    {"q":"Does homework widen or narrow the socioeconomic achievement gap?",
     "gt_views":["homework widens gap higher-SES families resources parental support","homework narrows gap additional practice time enrichment","format teacher follow-up mediate gap effects"],
     "gt_contradictions":[("widens gap SES resources","narrows gap via practice"),("format mediates","direct SES mechanism")],"gt_conf":"low"},
    {"q":"What is the optimal homework duration per grade level?",
     "gt_views":["10-minute rule per grade level moderate empirical support","time-based guidelines poor proxy quality over quantity","diminishing negative returns beyond individual thresholds"],
     "gt_contradictions":[("10-minute rule supported","time-based guidelines poor"),("diminishing returns","linear benefit")],"gt_conf":"medium"},
    {"q":"Is the homework-achievement correlation causal or merely correlational?",
     "gt_views":["experimental quasi-experimental evidence supports causal effect","observational research reverse causation higher achievers do more homework","natural experiment instrumental variable weaker no causal effects"],
     "gt_contradictions":[("causal from experiments","reverse causation threat"),("natural experiments null","experimental positive")],"gt_conf":"low"},
    {"q":"Do different homework types produce different learning outcomes?",
     "gt_views":["practice homework reinforces skills strongest effects achievement","project-based creative homework higher-order thinking skills","digital interactive homework comparable superior paper feedback"],
     "gt_contradictions":[("practice superior","project higher-order skills"),("digital comparable","practice strongest")],"gt_conf":"medium"},
    {"q":"How have homework research findings changed over recent decades?",
     "gt_views":["effect sizes declined 1980s recent decades publication bias","stronger causal identification weaker estimates methodological improvement","evidence base improved quality mixed direction no temporal trend"],
     "gt_contradictions":[("declining effect sizes","stable mixed evidence"),("methodological improvement","substantive change")],"gt_conf":"low"},
    {"q":"Does homework improve academic achievement in elementary school?",
     "gt_views":["near-zero effects elementary school homework Cooper meta-analysis","some positive practice effects structured elementary assignments","counterproductive for children under grade 3 developmental"],
     "gt_contradictions":[("near-zero null elementary","positive elementary structured")],"gt_conf":"low"},
    {"q":"Does homework quality matter more than quantity?",
     "gt_views":["assignment quality mediates effectiveness more than quantity","completion rate predicts achievement better than assigned time","teacher feedback on homework doubles achievement gain"],
     "gt_contradictions":[("quality over quantity","time-based assignment tradition")],"gt_conf":"medium"},
    {"q":"Does homework reduce student leisure time harmfully?",
     "gt_views":["homework crowds out leisure physical activity sleep","leisure time reduction compensated by skill development","moderate homework does not significantly reduce leisure time"],
     "gt_contradictions":[("crowds out leisure harmful","leisure compensated by learning")],"gt_conf":"medium"},
    {"q":"Is the 10-minute-per-grade homework rule evidence-based?",
     "gt_views":["10-minute rule commonly endorsed moderate research support","rule is arbitrary threshold lacks controlled experimental backing","optimal time varies by subject student quality not grade"],
     "gt_contradictions":[("10-minute rule supported empirically","rule arbitrary lacks evidence")],"gt_conf":"medium"},
    {"q":"Do students from low-income families benefit less from homework?",
     "gt_views":["low-income students derive smaller benefits lack resources quiet space","equivalent homework time different outcomes resource gap","uniform homework policies distributionally regressive equity concern"],
     "gt_contradictions":[("equal benefit regardless income","smaller benefit low-income resource")],"gt_conf":"low"},
    {"q":"Does homework completion rate predict academic success?",
     "gt_views":["completion rate predicts achievement independently prior ability","completion conflated with academic conscientiousness not homework itself","teacher enforcement mediates completion-achievement relationship"],
     "gt_contradictions":[("completion predicts achievement","completion conflated conscientiousness")],"gt_conf":"medium"},
    {"q":"Does digital homework differ in effectiveness from paper homework?",
     "gt_views":["digital homework faster feedback comparable superior outcomes","paper homework allows deeper processing less distraction","digital platforms increase completion not learning depth"],
     "gt_contradictions":[("digital superior feedback","paper deeper processing")],"gt_conf":"low"},
    {"q":"Do school districts banning homework see improved outcomes?",
     "gt_views":["homework bans no detectable negative achievement effects K-2","banning homework reduces family stress maintains achievement","homework ban reduces equity gap but mixed achievement evidence"],
     "gt_contradictions":[("homework ban neutral achievement","homework ban negative preparation")],"gt_conf":"low"},
    {"q":"Does homework help students with learning disabilities?",
     "gt_views":["homework harmful for students with learning disabilities cognitive load","accommodated homework beneficial special needs students","homework widens gap between typical and learning-disabled students"],
     "gt_contradictions":[("accommodated homework beneficial","standard homework harmful learning disabilities")],"gt_conf":"low"},
    {"q":"Is student-choice homework more effective?",
     "gt_views":["student autonomy choice homework improves intrinsic motivation","choice-based homework harder to align with curriculum goals","no consistent effectiveness difference choice versus assigned homework"],
     "gt_contradictions":[("choice improves motivation effectiveness","no consistent difference choice")],"gt_conf":"low"},
    {"q":"Does homework cause family stress and conflict?",
     "gt_views":["high homework loads cause family conflict parental frustration","family stress from homework peaks in high-achieving school contexts","homework conflict diminishes with student age and independence"],
     "gt_contradictions":[("homework causes family conflict","homework minimal family impact")],"gt_conf":"medium"},
    {"q":"Does international research support homework effectiveness?",
     "gt_views":["PISA data shows diminishing returns above 90 minutes per day","Asian educational systems high homework high performance association","cross-national variation suggests cultural context moderates homework effects"],
     "gt_contradictions":[("international positive homework-performance","PISA diminishing returns high loads")],"gt_conf":"medium"},
    {"q":"Does homework promote academic intrinsic motivation?",
     "gt_views":["poorly designed homework undermines intrinsic motivation younger students","homework can build self-efficacy through mastery experience","graded homework promotes extrinsic motivation reduces intrinsic"],
     "gt_contradictions":[("homework builds motivation self-efficacy","homework undermines intrinsic motivation")],"gt_conf":"low"},
    {"q":"Does homework frequency or amount matter most?",
     "gt_views":["frequency of homework more important than total volume","total time on task drives learning not frequency","frequency-amount interaction determines effectiveness neither alone"],
     "gt_contradictions":[("frequency more important amount","total time on task critical")],"gt_conf":"low"},
    {"q":"What role does teacher feedback play in homework effectiveness?",
     "gt_views":["teacher feedback on homework doubles achievement gain","collected-only homework produces negligible achievement effects","formative feedback homework benefits exceed summative grading"],
     "gt_contradictions":[("feedback doubles achievement","no feedback negligible effect")],"gt_conf":"medium"},
    {"q":"Does homework improve long-term knowledge retention?",
     "gt_views":["spaced practice homework improves long-term retention memory","most homework studies measure immediate not long-term retention","retention effects disappear without subsequent retrieval practice"],
     "gt_contradictions":[("homework improves long-term retention","retention effects not demonstrated")],"gt_conf":"low"},
    {"q":"Does homework reinforce or confuse classroom learning?",
     "gt_views":["practice homework reinforces previously taught material effectively","homework introducing new material causes confusion without teacher support","homework confusion correlates with low teacher explanation quality"],
     "gt_contradictions":[("homework reinforces classroom learning","homework new material causes confusion")],"gt_conf":"medium"},
    {"q":"Are teachers' beliefs about homework effectiveness accurate?",
     "gt_views":["teacher beliefs about homework benefits exceed evidence","teachers assign homework based on tradition professional norm not research","teacher homework design quality improves with professional development"],
     "gt_contradictions":[("teacher beliefs accurate evidence-based","teacher beliefs exceed evidence tradition")],"gt_conf":"medium"},
    {"q":"Does homework harm students with high extracurricular commitments?",
     "gt_views":["homework time conflicts with sports arts activities harm wellbeing","total after-school demand not homework alone determines stress","students with high extracurricular still benefit from moderate homework"],
     "gt_contradictions":[("homework harms extracurricular students","moderate homework benefits all students")],"gt_conf":"low"},
    {"q":"Is homework a form of educational inequality?",
     "gt_views":["homework advantage high-SES students resources technology space","homework deepens class-based inequality by rewarding home advantages","homework can be equalized with in-school support structures"],
     "gt_contradictions":[("homework deepens inequality","homework equalized school support")],"gt_conf":"low"},
    {"q":"Does grading homework improve student performance?",
     "gt_views":["graded homework improves completion accountability achievement","ungraded homework with feedback more effective than graded without","grading homework shifts motivation extrinsic reduces authentic learning"],
     "gt_contradictions":[("graded homework improves performance","grading reduces authentic learning")],"gt_conf":"low"},
    {"q":"Do high-performing school systems assign more homework?",
     "gt_views":["PISA top performers vary widely in homework load no clear pattern","Finland low homework high performance counterexample","homework volume positively correlates school performance confounded by SES"],
     "gt_contradictions":[("high performers assign more homework","top performers low homework Finland")],"gt_conf":"medium"},
    {"q":"Does homework affect student sleep duration?",
     "gt_views":["excessive homework reduces sleep by 30-60 minutes high schools","sleep reduction from homework impairs next-day learning consolidation","moderate homework does not significantly affect sleep duration"],
     "gt_contradictions":[("homework reduces sleep impairs","moderate homework no sleep effect")],"gt_conf":"medium"},
    {"q":"Should homework be eliminated in primary school?",
     "gt_views":["homework elimination primary school no negative achievement effects","some structured reading practice beneficial primary students","homework elimination reduces equity gap primary school context"],
     "gt_contradictions":[("eliminate homework primary neutral","primary homework some benefit structured")],"gt_conf":"low"},
    {"q":"Does homework benefit students of all ability levels equally?",
     "gt_views":["higher-ability students benefit more from homework challenge","lower-ability students benefit less without support structures","ability-differentiated homework reduces between-group achievement gap"],
     "gt_contradictions":[("all ability levels equal benefit","higher ability greater benefit homework")],"gt_conf":"low"},
    {"q":"How does homework policy vary across countries?",
     "gt_views":["vast cross-national variation homework hours policy not standardized","countries with explicit homework reduction policies see mixed results","no universal optimal homework policy emerges from comparative evidence"],
     "gt_contradictions":[("consistent optimal policy exists","vast variation no universal policy")],"gt_conf":"low"},
    {"q":"Does homework improve student academic engagement?",
     "gt_views":["homework positively predicts academic engagement school connection","mandatory homework can reduce engagement through perceived irrelevance","engagement effects depend on homework design and student choice"],
     "gt_contradictions":[("homework improves engagement","homework reduces engagement irrelevance")],"gt_conf":"low"},
    {"q":"Does student completion of homework predict future academic success?",
     "gt_views":["homework completion habits predict post-secondary academic success","completion habit reflects pre-existing conscientiousness not homework effect","completion-success link mediated by self-regulation not homework per se"],
     "gt_contradictions":[("completion predicts future success","completion reflects prior conscientiousness")],"gt_conf":"medium"},
    {"q":"Is there evidence for homework effects in randomized experiments?",
     "gt_views":["randomized studies show smaller positive effects than correlational","few true RCTs on homework ethical practical constraints","available RCT evidence supports modest positive effects practice homework"],
     "gt_contradictions":[("RCT evidence positive modest","correlational evidence inflated RCT smaller")],"gt_conf":"low"},
    {"q":"Does assigning less homework improve student wellbeing?",
     "gt_views":["homework reduction significantly improves wellbeing and stress","wellbeing gains from homework reduction not sustained long term","homework reduction improves wellbeing without academic achievement loss"],
     "gt_contradictions":[("homework reduction improves wellbeing sustainably","wellbeing gains not sustained long term")],"gt_conf":"medium"},
    {"q":"Do students perceive homework as valuable?",
     "gt_views":["students perceive practice homework as valuable when purposeful","most students view homework as burdensome not meaningful","student perceived value increases with teacher explanation of purpose"],
     "gt_contradictions":[("students value purposeful homework","students view homework burdensome")],"gt_conf":"low"},
    {"q":"Does homework develop study skills transferable to college?",
     "gt_views":["independent homework practice develops college-level study skills","homework in school differs fundamentally from college self-directed study","study skill transfer from homework to college not well evidenced"],
     "gt_contradictions":[("homework develops college skills","homework different college study no transfer")],"gt_conf":"low"},
    {"q":"Is spaced practice homework more effective than massed practice?",
     "gt_views":["spaced practice principle supports distributed homework over massed","most homework assigned massed near deadlines not spaced","spaced homework improves retention 30-50 percent versus massed"],
     "gt_contradictions":[("spaced homework superior retention","massed homework common practice not worse")],"gt_conf":"medium"},
    {"q":"Does homework in mathematics work differently than in reading?",
     "gt_views":["math homework shows stronger consistent effects than reading","reading homework effect depends on reading level text complexity","subject-specific effects suggest domain-differentiated homework policies"],
     "gt_contradictions":[("math homework stronger effects","reading homework comparable effects")],"gt_conf":"low"},
    {"q":"Can homework assignment apps improve effectiveness?",
     "gt_views":["app-based homework increases completion rates immediate feedback","technology homework same limitations as paper if design poor","apps improve equity through accessibility not inherently more effective"],
     "gt_contradictions":[("apps improve homework effectiveness","apps same design limitations as paper")],"gt_conf":"low"},
    {"q":"What is the evidence on homework and student test scores?",
     "gt_views":["homework positively associated standardized test scores secondary","association inflated by prior achievement confounding test preparation","controlled designs show smaller associations test scores homework"],
     "gt_contradictions":[("homework improves test scores","association confounded prior achievement")],"gt_conf":"medium"},
    {"q":"Does homework affect student social development?",
     "gt_views":["homework reduces peer social time may harm social development","social reduction from homework offset by collaborative homework","homework-social development tradeoff minimal moderate amounts"],
     "gt_contradictions":[("homework harms social development","homework social impact minimal moderate")],"gt_conf":"low"},
    {"q":"Do teachers assign homework because of professional norms?",
     "gt_views":["homework assigned based on tradition professional expectation not evidence","school culture and parent expectations drive homework norms","research-informed homework policies rare despite evidence availability"],
     "gt_contradictions":[("evidence-based homework assignment","tradition and norms drive homework")],"gt_conf":"medium"},
    {"q":"Does student homework autonomy predict academic outcomes?",
     "gt_views":["autonomous homework motivation predicts academic achievement independently","autonomy effect on homework mediated through self-efficacy","homework autonomy increases with age academic benefits accumulate"],
     "gt_contradictions":[("autonomy strongly predicts outcomes","autonomy mediated not direct effect")],"gt_conf":"low"},
    {"q":"What does the most rigorous evidence on homework show?",
     "gt_views":["most rigorous evidence shows small conditional positive effects","rigorous designs converge on smaller effects than correlational","no strong evidence base for current homework practices exists"],
     "gt_contradictions":[("rigorous evidence small positive effects","no strong evidence current practices")],"gt_conf":"medium"},
  ],

  "statins": [
    {"q":"Do statins prevent cardiovascular events in primary prevention?",
     "gt_views":["RCTs JUPITER WOSCOPS significant cardiovascular event reduction","absolute risk reduction small NNT 50-200 low-risk individuals","benefits highly heterogeneous by baseline cardiovascular risk"],
     "gt_contradictions":[("RCT evidence of benefit","small absolute NNT unfavorable"),("heterogeneous benefit","uniform benefit framing")],"gt_conf":"medium"},
    {"q":"Do statins increase the risk of type 2 diabetes?",
     "gt_views":["statins increase diabetes risk 9-12 percent high-intensity greater","cardiovascular benefits substantially outweigh diabetogenic risk","diabetes risk class effect insulin resistance prescribing decisions"],
     "gt_contradictions":[("diabetes risk confirmed meta-analysis","benefits outweigh diabetes risk"),("class effect mechanism","incidental artifact")],"gt_conf":"medium"},
    {"q":"Do statins cause muscle damage and myopathy?",
     "gt_views":["statin myopathy spectrum CK elevation rhabdomyolysis dose-dependent","myalgia largely nocebo-driven blinded RCT lower rates","muscle symptoms cause significant adherence reduction practice"],
     "gt_contradictions":[("spectrum myopathy real","nocebo-driven myalgia blinded"),("adherence harm real","nocebo underestimates")],"gt_conf":"medium"},
    {"q":"Do statins affect cognitive function or dementia risk?",
     "gt_views":["observational studies statins reduce dementia Alzheimer risk","RCT PROSPER HPS no cognitive benefit found","case reports memory impairment cognitive side effects disputed causality"],
     "gt_contradictions":[("observational dementia benefit","RCT no cognitive benefit"),("cognitive side effects","observational benefit")],"gt_conf":"low"},
    {"q":"Is statin therapy effective for women in primary prevention?",
     "gt_views":["RCT subgroup analyses similar relative risk reductions women men","women underrepresented primary prevention NNT too high","sex-specific analyses comparable relative risk different absolute rate"],
     "gt_contradictions":[("similar relative RR women men","lower baseline NNT unfavorable women"),("underrepresentation limits","available data supports women")],"gt_conf":"low"},
    {"q":"Do statins have anti-inflammatory effects beyond cholesterol lowering?",
     "gt_views":["pleiotropic anti-inflammatory effects CRP endothelial protection","cardiovascular benefit primarily attributable LDL-C reduction pleiotropic secondary","JUPITER CRP reduction predicted benefit LDL remains primary driver"],
     "gt_contradictions":[("pleiotropic effects meaningful","LDL lowering primary"),("JUPITER supports inflammation","LDL change primary driver")],"gt_conf":"medium"},
    {"q":"Do statins reduce all-cause mortality in low-risk populations?",
     "gt_views":["meta-analyses primary prevention significant all-cause mortality reduction","individual trials low-risk no significant all-cause mortality reduction","benefit depends trial duration baseline risk underpowered short trials"],
     "gt_contradictions":[("meta-analytic mortality benefit","individual trial non-significant"),("underpowered trials","sufficient power meta-analysis")],"gt_conf":"low"},
    {"q":"Are there reliable non-pharmaceutical alternatives to statins?",
     "gt_views":["lifestyle diet exercise smoking cessation effective first-line low-risk","statins irreplaceable high-risk patients lifestyle cannot match","plant sterols omega-3 Mediterranean diet substitute statin-intolerant"],
     "gt_contradictions":[("lifestyle first-line low-risk","statins irreplaceable high-risk"),("dietary alternatives effective","supplementary only framing")],"gt_conf":"medium"},
    {"q":"Does statin benefit depend on baseline LDL-C level?",
     "gt_views":["proportional risk reduction consistent across baseline LDL range","absolute risk reduction greater higher baseline LDL cardiovascular risk","LDL hypothesis predicts benefit any baseline absolute justifies treatment"],
     "gt_contradictions":[("proportional benefit regardless baseline","absolute greater at high LDL"),("LDL hypothesis uniform","risk-stratified prescribing needed")],"gt_conf":"medium"},
    {"q":"How should statin therapy be monitored in clinical practice?",
     "gt_views":["LDL-C monitoring guides dose titration to evidence-based targets","treat-to-target not superior fixed-dose high-intensity approach","adherence tolerability more important than specific LDL targets"],
     "gt_contradictions":[("treat-to-LDL target","fixed-dose high intensity"),("adherence over targets","target-based titration")],"gt_conf":"medium"},
    {"q":"What is the absolute risk reduction from statins in primary prevention?",
     "gt_views":["NNT ranges 80-200 five years major cardiovascular event prevention","absolute reduction modest 0.5-1.5 percent annual event rate","absolute benefit weighed against side effect costs annually"],
     "gt_contradictions":[("modest absolute benefit NNT high","substantial prevention justifies treatment")],"gt_conf":"medium"},
    {"q":"Do high-intensity statins provide greater benefit than moderate-intensity?",
     "gt_views":["high-intensity statins further reduce LDL and cardiovascular events","high-intensity increases side effects diabetes myopathy risk","moderate-intensity optimal balance efficacy safety most patients"],
     "gt_contradictions":[("high-intensity greater benefit","moderate-intensity optimal safety efficacy balance")],"gt_conf":"medium"},
    {"q":"Should statins be prescribed to patients with LDL below 100 mg/dL?",
     "gt_views":["statin benefit proportional LDL lowering regardless starting point","low LDL patients absolute risk low treatment threshold not justified","cardiovascular risk score not LDL alone should drive prescription"],
     "gt_contradictions":[("benefit regardless starting LDL","low LDL threshold not justified")],"gt_conf":"low"},
    {"q":"Do statins reduce stroke risk as well as coronary heart disease?",
     "gt_views":["statins reduce ischemic stroke risk 20-25 percent primary prevention","coronary heart disease reduction stronger than stroke in trials","hemorrhagic stroke may increase with intensive LDL lowering"],
     "gt_contradictions":[("statins reduce ischemic stroke","hemorrhagic stroke risk increase intensive")],"gt_conf":"medium"},
    {"q":"Are statins over-prescribed in low-risk populations?",
     "gt_views":["current guidelines overly liberal primary prevention over-prescribing","NNT too high low-risk prescribing not cost-effective","over-prescribing leads to nocebo side effects unnecessary adherence burden"],
     "gt_contradictions":[("statins appropriately prescribed guideline-concordant","over-prescribed low-risk NNT high")],"gt_conf":"medium"},
    {"q":"Can statins be safely stopped in elderly patients?",
     "gt_views":["statin discontinuation elderly increased cardiovascular events observational","deprescribing statins frail elderly may reduce pill burden safely","insufficient RCT evidence for or against stopping statins over 80"],
     "gt_contradictions":[("statin discontinuation increases events","deprescribing frail elderly safe reasonable")],"gt_conf":"low"},
    {"q":"Do statins prevent cancer?",
     "gt_views":["observational studies suggest statins may reduce several cancer types","RCT evidence no consistent cancer prevention effect statins","statin-cancer association likely confounded healthy user bias"],
     "gt_contradictions":[("statins reduce cancer observational","RCT no cancer prevention")],"gt_conf":"low"},
    {"q":"Do statins have different efficacy across racial or ethnic groups?",
     "gt_views":["statin efficacy broadly similar across racial ethnic groups trials","Asian populations require lower statin doses pharmacogenomic differences","Black patients underrepresented trials uncertainty race-specific benefit"],
     "gt_contradictions":[("consistent efficacy across racial groups","Asian lower dose pharmacogenomic")],"gt_conf":"low"},
    {"q":"Should statin therapy guidelines use absolute or relative risk?",
     "gt_views":["absolute risk reduction more informative patient decision-making","relative risk reduction overstates benefit low-risk populations","guidelines should present both absolute relative communicate uncertainty"],
     "gt_contradictions":[("absolute risk presentation needed","relative risk standard practice guidelines")],"gt_conf":"medium"},
    {"q":"What does Cochrane evidence show about statin primary prevention?",
     "gt_views":["Cochrane reviews support statin primary prevention modest benefit","Taylor 2013 Cochrane found significant reduction events all-cause mortality","Cochrane reviews criticized for industry-funded trial inclusion bias"],
     "gt_contradictions":[("Cochrane supports statin primary prevention","Cochrane industry bias limitations")],"gt_conf":"medium"},
    {"q":"Do statins reduce heart failure hospitalizations?",
     "gt_views":["statins reduce coronary-related heart failure events secondary","no clear evidence statins prevent non-ischemic heart failure","CORONA GISSI-HF trials showed no mortality benefit heart failure"],
     "gt_contradictions":[("statins reduce heart failure coronary","CORONA GISSI no mortality benefit heart failure")],"gt_conf":"low"},
    {"q":"Do statins help patients with chronic kidney disease?",
     "gt_views":["statins reduce cardiovascular events CKD patients without dialysis","dialysis patients no significant statin cardiovascular benefit 4D AURORA","CKD patients higher absolute risk statin benefit substantial pre-dialysis"],
     "gt_contradictions":[("statins benefit CKD pre-dialysis","dialysis no benefit 4D AURORA")],"gt_conf":"medium"},
    {"q":"Does statin adherence affect cardiovascular outcomes?",
     "gt_views":["adherence below 80 percent significantly attenuates statin benefit","discontinuation within 6 months associated with major event increase 46 percent","real-world statin effectiveness substantially lower than trial efficacy"],
     "gt_contradictions":[("adherence critical outcome determinant","real-world effectiveness lower trial")],"gt_conf":"medium"},
    {"q":"Do statins reduce atrial fibrillation risk?",
     "gt_views":["observational studies suggest statins reduce atrial fibrillation incidence","RCT evidence no consistent atrial fibrillation prevention from statins","anti-inflammatory mechanism plausible but not demonstrated in trials"],
     "gt_contradictions":[("observational statins reduce AF","RCT no consistent AF prevention")],"gt_conf":"low"},
    {"q":"What is the cost-effectiveness of statins in primary prevention?",
     "gt_views":["generic statins cost-effective primary prevention high-risk patients","low-risk patients statins not cost-effective quality-adjusted life year threshold","cost-effectiveness varies dramatically by country healthcare system drug price"],
     "gt_contradictions":[("statins cost-effective primary prevention","low-risk not cost-effective QALY threshold")],"gt_conf":"medium"},
    {"q":"Do statins interact harmfully with other common medications?",
     "gt_views":["statins interact CYP3A4 drugs increase myopathy risk amiodarone","drug interactions clinically significant require dose adjustment monitoring","most common drug combinations statin safe routine monitoring sufficient"],
     "gt_contradictions":[("significant drug interactions require adjustment","most combinations safe routine monitoring")],"gt_conf":"medium"},
    {"q":"Should patients with statin side effects switch or stop therapy?",
     "gt_views":["switching to different statin resolves myopathy 70-80 percent cases","nocebo education rechallenge succeeds most patients with intolerance","stopping statins high-risk patients causes cardiovascular harm outweighs benefit"],
     "gt_contradictions":[("switching resolves side effects success","stopping necessary some patients unavoidable")],"gt_conf":"medium"},
    {"q":"Does CoQ10 supplementation prevent statin side effects?",
     "gt_views":["CoQ10 supplementation plausible mechanism statin myopathy ubiquinone pathway","RCT evidence CoQ10 inconsistent benefit statin myalgia","CoQ10 commonly recommended clinical practice without strong evidence base"],
     "gt_contradictions":[("CoQ10 mechanism plausible prevention","RCT evidence CoQ10 inconsistent benefit")],"gt_conf":"low"},
    {"q":"Do statins benefit patients with metabolic syndrome?",
     "gt_views":["metabolic syndrome high cardiovascular risk statins clearly beneficial","metabolic syndrome diagnosis improves statin targeting beyond LDL","lifestyle modification primary treatment metabolic syndrome before statin"],
     "gt_contradictions":[("statins beneficial metabolic syndrome","lifestyle before statin metabolic syndrome")],"gt_conf":"medium"},
    {"q":"Should statins be used in children with familial hypercholesterolemia?",
     "gt_views":["familial hypercholesterolemia statins safe effective children age 8-10","long-term pediatric statin safety unclear lifelong exposure concern","dietary treatment first pediatric FH before statin age thresholds"],
     "gt_contradictions":[("statins safe effective pediatric FH","long-term safety unclear pediatric concern")],"gt_conf":"medium"},
    {"q":"What is the evidence that statins reduce dementia risk?",
     "gt_views":["observational cohort studies show 30-40 percent dementia reduction statins","PROSPER trial no cognitive benefit 3-5 year follow-up statin","dementia prevention requires decades not captured in statin trials"],
     "gt_contradictions":[("observational dementia reduction","PROSPER no cognitive benefit trial")],"gt_conf":"low"},
    {"q":"Do statins help prevent recurrent stroke?",
     "gt_views":["SPARCL trial atorvastatin reduces stroke recurrence after ischemic stroke","high-intensity statin slightly increases hemorrhagic stroke risk post-ischemic","net benefit stroke recurrence prevention favorable most patients SPARCL"],
     "gt_contradictions":[("SPARCL reduces stroke recurrence","hemorrhagic stroke risk increase high-intensity")],"gt_conf":"medium"},
    {"q":"What is the WOSCOPS trial evidence for long-term statin benefit?",
     "gt_views":["WOSCOPS 20-year follow-up shows durable mortality benefit pravastatin","long-term follow-up benefit persists after statin discontinuation legacy effect","WOSCOPS population high-risk Scottish men not generalizable contemporary"],
     "gt_contradictions":[("WOSCOPS durable long-term benefit","WOSCOPS not generalizable contemporary population")],"gt_conf":"medium"},
    {"q":"Are statins indicated for patients with normal cholesterol and high CRP?",
     "gt_views":["statin benefit proportional LDL lowering regardless starting cholesterol","patients normal cholesterol low absolute cardiovascular risk NNT high","JUPITER rosuvastatin benefit low-LDL high-CRP supports broad use"],
     "gt_contradictions":[("statin benefit normal cholesterol JUPITER","low cholesterol NNT too high")],"gt_conf":"medium"},
    {"q":"Do plant-based diets reduce the need for statin therapy?",
     "gt_views":["plant-based diets reduce LDL by 10-15 percent reducing statin need","dietary LDL reduction insufficient to replace statin therapy high-risk","lifestyle optimization reduces cardiovascular risk statin-additive not alternative"],
     "gt_contradictions":[("plant-based diet reduces statin need","dietary change insufficient statin alternative")],"gt_conf":"medium"},
    {"q":"Do statins affect bone density or fracture risk?",
     "gt_views":["observational studies suggest statins increase bone density reduce fractures","RCT evidence does not confirm bone protective statin effect","statin bone benefit likely confounded healthy user bias observational"],
     "gt_contradictions":[("statins increase bone density reduce fracture","RCT no bone protective effect")],"gt_conf":"low"},
    {"q":"What is the evidence for statins in patients with established coronary artery disease?",
     "gt_views":["secondary prevention statins clearly indicated established CAD NNT 20","high-intensity statin targets LDL under 70 secondary prevention","4S LIPID trials established statin mortality benefit secondary prevention"],
     "gt_contradictions":[("secondary prevention clear benefit","benefit depends patient frailty comorbidity")],"gt_conf":"medium"},
    {"q":"Do statins reduce perioperative cardiovascular risk?",
     "gt_views":["perioperative statin continuation reduces cardiac events surgical patients","statin withdrawal perioperatively associated with rebound inflammation events","evidence perioperative statin initiation mixed naive patients"],
     "gt_contradictions":[("perioperative continuation reduces risk","initiation naive perioperative mixed evidence")],"gt_conf":"low"},
    {"q":"What are the ethical issues in statin over-prescription?",
     "gt_views":["over-prescribing statins medicalizes healthy lifestyle concerns nocebo","informed consent requires absolute risk communication not only relative","industry relationships bias guideline committee statin recommendations upward"],
     "gt_contradictions":[("statin guidelines appropriate evidence-based","industry bias inflates recommendations over-prescribing")],"gt_conf":"medium"},
    {"q":"What does precision medicine offer for future statin prescribing?",
     "gt_views":["pharmacogenomic testing identifies statin metabolism high-risk patients","polygenic risk scores improve cardiovascular risk stratification beyond LDL","precision medicine statin prescribing not yet clinically implemented broadly"],
     "gt_contradictions":[("pharmacogenomics improves statin prescribing","precision medicine not implemented clinical practice")],"gt_conf":"low"},
    {"q":"Do statins have long-term safety beyond 10 years?",
     "gt_views":["long-term statin use trials 20 years show sustained safety","very long-term cognitive effects cancer risk uncertain beyond trials","accumulated evidence supports decade-long statin safety secondary prevention"],
     "gt_contradictions":[("long-term safety demonstrated trials","very long-term effects uncertain beyond trials")],"gt_conf":"low"},
    {"q":"Do statins reduce erectile dysfunction?",
     "gt_views":["statins improve erectile dysfunction endothelial mechanism cardiovascular","some case reports testosterone reduction sexual side effects statins","RCT evidence no significant sexual dysfunction from statin therapy"],
     "gt_contradictions":[("statins improve erectile function","sexual side effects testosterone reduction")],"gt_conf":"low"},
    {"q":"Is there evidence statins prevent venous thromboembolism?",
     "gt_views":["JUPITER trial rosuvastatin reduced DVT pulmonary embolism risk","subsequent studies conflicting findings statin VTE prevention","VTE prevention not established indication for statin therapy"],
     "gt_contradictions":[("JUPITER statins reduce VTE","conflicting findings no established indication")],"gt_conf":"low"},
    {"q":"Do patients on statins still need dietary fat reduction?",
     "gt_views":["statins reduce LDL regardless diet dietary change still recommended","diet modification statin-additive provides complementary cardiovascular risk reduction","statin effectiveness reduces motivation dietary adherence behavioral concern"],
     "gt_contradictions":[("diet still important statin additive","statin reduces motivation dietary change")],"gt_conf":"low"},
    {"q":"Do statins affect the gut microbiome?",
     "gt_views":["statins alter gut microbiome composition potentially beneficially","microbiome effects of statins confound cardiovascular outcome interpretation","statin-microbiome interaction not clinically actionable current evidence"],
     "gt_contradictions":[("statins alter microbiome beneficially","microbiome evidence not clinically actionable")],"gt_conf":"low"},
    {"q":"Should statin therapy target LDL or total cardiovascular risk?",
     "gt_views":["total cardiovascular risk superior target to LDL alone for prescribing","LDL-C reduction per mmol primary surrogate endpoint statin trials","risk calculator uncertainty undermines risk-based prescribing reliability"],
     "gt_contradictions":[("total risk superior target to LDL","LDL primary surrogate statin trials")],"gt_conf":"medium"},
    {"q":"Do statins have a role in preventing contrast-induced nephropathy?",
     "gt_views":["statins may protect kidney from contrast nephropathy oxidative stress","small studies show statin pretreatment reduces contrast nephropathy incidence","meta-analysis evidence mixed not sufficient for routine indication"],
     "gt_contradictions":[("statins protect contrast nephropathy","meta-analysis mixed not sufficient indication")],"gt_conf":"low"},
    {"q":"What is the comparative effectiveness of different statin types?",
     "gt_views":["atorvastatin rosuvastatin most potent LDL reduction high-intensity","simvastatin lovastatin lower potency more drug interactions","potency differences translate to clinical outcome differences in comparative trials"],
     "gt_contradictions":[("potency differences translate clinical outcomes","bioequivalent outcomes within intensity class")],"gt_conf":"low"},
    {"q":"Do statins affect telomere length and biological aging?",
     "gt_views":["observational evidence statins associated with longer telomere length aging","no established mechanism statins directly protect telomere biology","telomere associations confounded by cardiovascular health proxy selection bias"],
     "gt_contradictions":[("statins protect telomere length aging","telomere associations confounded selection bias")],"gt_conf":"low"},
    {"q":"What is the evidence for statins in patients with high lipoprotein(a)?",
     "gt_views":["elevated lipoprotein(a) increases residual cardiovascular risk beyond LDL statins","statins do not lower Lp(a) levels alternative agents PCSK9 inhibitors needed","high Lp(a) patients benefit from intensive statin lowering LDL even without Lp(a) reduction"],
     "gt_contradictions":[("statins do not lower Lp(a) alternative agents needed","statins benefit high Lp(a) through LDL lowering")],"gt_conf":"low"},
  ],

  "minwage": [
    {"q":"Does raising the minimum wage increase unemployment?",
     "gt_views":["standard competitive theory predicts disemployment small negative employment effects","Card Krueger quasi-experimental no disemployment effect minimum wage","employment effects heterogeneous monopsony power can increase employment"],
     "gt_contradictions":[("disemployment from theory","no effect quasi-experimental"),("monopsony reverses sign","competitive market negative")],"gt_conf":"high"},
    {"q":"Do minimum wage increases reduce poverty rates?",
     "gt_views":["10 percent minimum wage increase reduces poverty rate 2-4 percent","minimum wages reduce income inequality among low-wage workers","poverty effects modest because many minimum wage workers not in poor households"],
     "gt_contradictions":[("minimum wage reduces poverty","modest poverty effect targeting problem")],"gt_conf":"high"},
    {"q":"Does monopsony explain the absence of employment effects?",
     "gt_views":["monopsony employer wage-setting power explains small minimum wage employment effects","monopsony empirically present low-wage labor markets hiring costs turnover","competitive model sufficient minimum wage employment effects small but present"],
     "gt_contradictions":[("monopsony explains no employment effect","competitive model negative effects present")],"gt_conf":"high"},
    {"q":"Does the Seattle minimum wage study show negative effects?",
     "gt_views":["Seattle minimum wage reduced hours worked low-wage workers 9 percent income 125 dollars","alternative Seattle study shows no significant employment loss restaurants","administrative payroll data shows hours cut employment counts understate response"],
     "gt_contradictions":[("Seattle hours cut income loss negative","Seattle alternative study no significant loss")],"gt_conf":"high"},
    {"q":"Do minimum wages affect youth employment specifically?",
     "gt_views":["teenagers employment elasticity negative 0.1 to 0.2 minimum wage","youth employment effects smaller modern studies than earlier estimates","youth disemployment concentrated very young workers not teens broadly"],
     "gt_contradictions":[("teen disemployment minimum wage negative","youth effects smaller modern studies")],"gt_conf":"high"},
    {"q":"Does the Card and Krueger study design hold up methodologically?",
     "gt_views":["Card Krueger NJ PA natural experiment robust to reanalysis","Neumark Wascher reanalysis payroll data contradicts Card Krueger findings","border discontinuity designs strengthen Card Krueger methodology confirmed"],
     "gt_contradictions":[("Card Krueger robust natural experiment","Neumark Wascher reanalysis contradicts")],"gt_conf":"high"},
    {"q":"Do minimum wage increases cause inflation?",
     "gt_views":["minimum wage increases raise restaurant food prices 0.7 percent 10 percent wage increase","price effects small localized not macroeconomic inflation concern","input cost pass-through to prices partially offsets employment demand effects"],
     "gt_contradictions":[("minimum wage causes price inflation","price effects small not macroeconomic concern")],"gt_conf":"medium"},
    {"q":"Does minimum wage reduce income inequality?",
     "gt_views":["minimum wages significantly reduce wage inequality bottom end distribution","Gini coefficient improvements from minimum wages substantial low-wage economies","minimum wage effects on inequality offset by employment losses inequality rises"],
     "gt_contradictions":[("minimum wages reduce inequality Gini","employment losses offset inequality reduction")],"gt_conf":"high"},
    {"q":"What does the border discontinuity research design show?",
     "gt_views":["Dube Lester Reich border discontinuity shows no employment loss restaurant retail","border design controls local economic trends prior studies fail isolate","border design criticized for failing account local minimum wage variation confound"],
     "gt_contradictions":[("border design shows no employment loss","border design criticized local confound")],"gt_conf":"high"},
    {"q":"Do small businesses suffer more from minimum wage increases?",
     "gt_views":["small businesses higher exposure exit more from minimum wage increases","large businesses absorb minimum wages through thin profit margins better","small business differential exit 15-20 percent labor demand response"],
     "gt_contradictions":[("small businesses greater harm exit","large businesses absorb better small differential")],"gt_conf":"medium"},
    {"q":"Does minimum wage increase affect restaurant employment?",
     "gt_views":["restaurant employment no detectable loss border discontinuity designs","restaurant employment negative effects 0.05-0.15 elasticity regression discontinuity","restaurant industry high exposure minimum wage largest affected sector"],
     "gt_contradictions":[("restaurant employment no detectable loss","restaurant negative effects 0.05-0.15 elasticity")],"gt_conf":"high"},
    {"q":"Does minimum wage reduce reliance on public assistance?",
     "gt_views":["minimum wages reduce food stamp Medicaid take-up low-income families","public assistance reduction substantial enough to offset fiscal minimum wage cost","public assistance reduction modest because minimum wage workers often ineligible"],
     "gt_contradictions":[("minimum wage reduces public assistance substantially","public assistance reduction modest ineligibility")],"gt_conf":"medium"},
    {"q":"What are the long-run versus short-run employment effects?",
     "gt_views":["long-run employment effects larger than short-run capital substitution automation","short-run employment effects small long-run adjustment produces larger disemployment","long-run effects difficult to identify confounded with technological change"],
     "gt_contradictions":[("long-run larger effects capital substitution","long-run effects difficult identify confounded")],"gt_conf":"high"},
    {"q":"Do minimum wages lead to automation and job displacement?",
     "gt_views":["minimum wage increases accelerate automation robot adoption low-wage industry","automation substitution limited because minimum wage tasks difficult to automate","automation response exists but accounts for small fraction employment response"],
     "gt_contradictions":[("minimum wage accelerates automation displacement","automation substitution limited difficult tasks")],"gt_conf":"medium"},
    {"q":"Is minimum wage more effective than EITC for poverty reduction?",
     "gt_views":["EITC better targeted poor households minimum wage benefits non-poor workers","minimum wage and EITC complementary different populations served","minimum wage superior EITC because doesn't require filing income eligibility"],
     "gt_contradictions":[("EITC better targeted poverty reduction","minimum wage and EITC complementary")],"gt_conf":"medium"},
    {"q":"Do women benefit more from minimum wage increases than men?",
     "gt_views":["women disproportionately represented minimum wage workers benefit more","minimum wage reduces gender wage gap low-wage industries","employment disemployment effects fall more on women high minimum wage exposure"],
     "gt_contradictions":[("women benefit more minimum wage","disemployment effects fall more on women")],"gt_conf":"medium"},
    {"q":"Does raising the minimum wage hurt small business owners?",
     "gt_views":["minimum wages reduce profits small businesses labor intensive sectors","small business survival rates drop modestly after large minimum wage increases","small business owners adapt through price increases efficiency gains survive"],
     "gt_contradictions":[("minimum wages harm small business profits","small businesses adapt survive price efficiency")],"gt_conf":"medium"},
    {"q":"What does the Congressional Budget Office say about minimum wage effects?",
     "gt_views":["CBO estimates 15 dollar minimum wage lifts 900k from poverty job loss 1.4 million","CBO uncertainty range substantial zero to 2.7 million job loss","CBO estimates broadly consistent with academic consensus moderate employment effects"],
     "gt_contradictions":[("CBO projects significant job loss 1.4 million","CBO uncertainty range zero jobs lost possible")],"gt_conf":"high"},
    {"q":"Do minimum wages affect self-employment rates?",
     "gt_views":["minimum wages increase self-employment as workers leave reduced-hour employment","minimum wages reduce self-employment as wage employment more attractive","minimum wage effect on self-employment ambiguous and context-dependent"],
     "gt_contradictions":[("minimum wage increases self-employment","minimum wage reduces self-employment")],"gt_conf":"low"},
    {"q":"Do minimum wages affect health outcomes for low-wage workers?",
     "gt_views":["minimum wage increases improve health outcomes stress food security low-income","minimum wage employment loss harms health through income reduction","health benefits minimum wage concentrated workers not losing jobs"],
     "gt_contradictions":[("minimum wage improves health outcomes","employment loss harms health income reduction")],"gt_conf":"medium"},
    {"q":"Do minimum wages affect crime rates?",
     "gt_views":["minimum wage increases reduce property crime through income effect","minimum wage unemployment increases crime through disemployment","crime effects of minimum wage uncertain confounded macroeconomic conditions"],
     "gt_contradictions":[("minimum wage reduces crime income effect","minimum wage unemployment increases crime")],"gt_conf":"low"},
    {"q":"Is minimum wage research subject to publication bias?",
     "gt_views":["publication bias inflates no-employment-effect findings meta-analysis","Doucouliagos Stanwick funnel plot asymmetry suggests publication bias both directions","Neumark Wascher no clear publication bias negative employment effects"],
     "gt_contradictions":[("publication bias inflates no-effect findings","no clear publication bias negative effects")],"gt_conf":"high"},
    {"q":"Do spillover effects of minimum wages reach workers above the minimum?",
     "gt_views":["minimum wages create wage spillovers 10-20 percent above minimum range","spillover effects reduce inequality beyond directly affected workers","spillover effects disappear once controlling for employment composition changes"],
     "gt_contradictions":[("spillovers reach workers above minimum","spillover effects disappear controlling composition")],"gt_conf":"medium"},
    {"q":"Does employer profit absorption reduce employment effects?",
     "gt_views":["employers absorb some minimum wage cost through profit margin compression","profit margin compression limited many low-wage industries already thin margins","profit compression and price increases together reduce employment effects"],
     "gt_contradictions":[("profit absorption reduces employment effects","thin margins limit profit absorption capacity")],"gt_conf":"medium"},
    {"q":"Do minimum wages affect working hours more than employment levels?",
     "gt_views":["employers reduce hours more than eliminate jobs responding to minimum wages","employment count measures miss hour reduction full impact minimum wage","Seattle study shows income fell despite employment count stable hours cut"],
     "gt_contradictions":[("hours reduced more than employment","employment counts sufficient measure response")],"gt_conf":"high"},
    {"q":"Does minimum wage affect urban and rural areas differently?",
     "gt_views":["rural areas smaller wage bite minimum wages smaller employment effects","urban labor markets closer to monopsony minimum wages larger effects","uniform federal minimum wage inappropriate given urban rural wage differences"],
     "gt_contradictions":[("rural smaller bite smaller effects","uniform federal minimum wage inappropriate")],"gt_conf":"medium"},
    {"q":"Do index-linked minimum wages produce better outcomes than fixed increases?",
     "gt_views":["indexed minimum wages reduce policy uncertainty enable business planning","regular small indexed increases less disruptive than large periodic jumps","indexed minimum wages remove political discretion may freeze inadequate base"],
     "gt_contradictions":[("indexed minimum wages better outcomes","indexed wages freeze inadequate base level")],"gt_conf":"low"},
    {"q":"What does the gig economy evidence say about minimum wage effects?",
     "gt_views":["gig workers excluded minimum wage coverage benefit from extension","gig economy expansion partly driven by minimum wage avoidance classification","minimum wages for gig workers reduce flexibility income volatility increases"],
     "gt_contradictions":[("gig workers benefit extension minimum wage","minimum wage gig extension reduces flexibility")],"gt_conf":"low"},
    {"q":"Do minimum wages affect minority and low-income business owners?",
     "gt_views":["minority business owners disproportionately hurt minimum wage increases labor costs","minimum wage reduces gap between minority white worker wages benefiting communities","net effect on minority communities depends employment versus wage effect balance"],
     "gt_contradictions":[("minority business owners hurt costs","minority workers benefit wage gap reduction")],"gt_conf":"medium"},
    {"q":"Is there macroeconomic stimulus from minimum wage increases?",
     "gt_views":["minimum wages increase aggregate demand through lower-income worker spending","macroeconomic stimulus from minimum wages offset by reduced business investment","minimum wage multiplier effects concentrated local labor markets limited aggregate"],
     "gt_contradictions":[("minimum wage increases aggregate demand","macroeconomic stimulus offset business investment reduction")],"gt_conf":"medium"},
    {"q":"What would a $25 minimum wage mean for the US labor market?",
     "gt_views":["25 dollar minimum wage unprecedented would produce larger employment effects","25 dollar minimum wage 100 percent median wage bite extremely high international","regional variation means 25 dollar minimum catastrophic rural appropriate urban"],
     "gt_contradictions":[("25 dollar minimum wage larger employment effects","25 dollar minimum appropriate urban high-wage areas")],"gt_conf":"high"},
    {"q":"Do minimum wages affect firm entry and exit rates?",
     "gt_views":["minimum wages reduce firm entry rates increase exit low-wage industries","firm turnover from minimum wage accounts 15-20 percent employment response","firm entry exit effects more important than within-firm employment reduction"],
     "gt_contradictions":[("minimum wages reduce entry increase exit","firm effects smaller than within-firm response")],"gt_conf":"medium"},
    {"q":"Does the minimum wage research consensus favor Card or Neumark?",
     "gt_views":["modern consensus closer to Card no large employment effects small negative","Neumark Wascher meta-analysis shows consistent negative employment effects","no consensus exists researchers disagree based on design philosophy choice"],
     "gt_contradictions":[("modern consensus favors Card small effects","Neumark meta-analysis consistent negative effects")],"gt_conf":"high"},
    {"q":"Do minimum wages affect income mobility for low-wage workers?",
     "gt_views":["minimum wages improve income mobility through wage floor higher rungs","minimum wages reduce mobility by trapping workers minimum wage jobs","mobility effects ambiguous depend on whether wage or employment effect dominates"],
     "gt_contradictions":[("minimum wage improves income mobility","minimum wage reduces mobility trap workers")],"gt_conf":"medium"},
    {"q":"What does cross-country evidence show about minimum wage employment effects?",
     "gt_views":["cross-country evidence shows high minimum wages relative median no large employment loss","OECD countries high minimum wage ratio low unemployment coexist","cross-country comparisons confounded by labor market institutions welfare states"],
     "gt_contradictions":[("cross-country high minimum no large employment loss","cross-country confounded institutions welfare states")],"gt_conf":"high"},
    {"q":"Should the minimum wage be set federally or at the state level?",
     "gt_views":["state-level minimum wages more appropriate local labor market conditions","federal minimum wage provides floor eliminates race-to-bottom between states","optimal minimum wage policy combines federal floor with state local flexibility"],
     "gt_contradictions":[("state-level minimum wages more appropriate local","federal minimum wage necessary race-to-bottom")],"gt_conf":"medium"},
    {"q":"Does the minimum wage reduce childhood poverty rates?",
     "gt_views":["minimum wage increase reduces childhood poverty rates 2-3 percent","childhood poverty reduction minimum wage concentrated non-poor working families","minimum wage effects on children poverty depend employment effects family structure"],
     "gt_contradictions":[("minimum wage reduces childhood poverty","childhood poverty reduction limited family targeting")],"gt_conf":"medium"},
    {"q":"Are minimum wage effects on employment symmetric with cuts?",
     "gt_views":["employment effects asymmetric downward minimum wage rigidity prevents wage cuts","employment response to increases different than decreases institutional differences","symmetric competitive market would predict equal response to equal increases cuts"],
     "gt_contradictions":[("minimum wage effects asymmetric institutional","symmetric competitive model equal response")],"gt_conf":"low"},
    {"q":"What are the macroeconomic effects of a large minimum wage increase?",
     "gt_views":["large minimum wage increases aggregate demand through consumption low-income","macroeconomic effects depend critically on employment response magnitude","macroeconomic simulations show modestly positive GDP effect small minimum wages"],
     "gt_contradictions":[("large minimum wage positive aggregate demand","macroeconomic effects depend employment magnitude")],"gt_conf":"medium"},
    {"q":"Do minimum wages affect the retail sector employment?",
     "gt_views":["retail employment sensitive minimum wages large fraction minimum wage workers","retail employment effects smaller than restaurant effects more capital flexibility","retail automation response minimum wage accelerates checkout replacement"],
     "gt_contradictions":[("retail employment sensitive minimum wages","retail effects smaller than restaurant sector")],"gt_conf":"medium"},
    {"q":"Is employer turnover reduction a significant offset to minimum wage costs?",
     "gt_views":["turnover reduction from higher minimum wages significant cost offset","Card Krueger identified turnover reduction as mechanism no employment effect","turnover reduction benefits smaller than wage cost increase for most employers"],
     "gt_contradictions":[("turnover reduction significant offset mechanism","turnover reduction smaller than wage cost increase")],"gt_conf":"medium"},
    {"q":"What minimum wage level would maximize welfare of low-income workers?",
     "gt_views":["optimal minimum wage 50-60 percent local median wage monopsony model","optimal minimum wage depends trade-off employment wage effects model assumptions","empirical evidence insufficient to specify welfare-maximizing minimum wage precisely"],
     "gt_contradictions":[("optimal minimum wage 50-60 percent median","empirical insufficient specify optimal precisely")],"gt_conf":"high"},
    {"q":"Do minimum wages affect unionization rates?",
     "gt_views":["minimum wages reduce incentive to unionize substitute for union wage floor","minimum wages complement union organizing raises floor both union nonunion","unionization rates and minimum wages move together political economy correlation"],
     "gt_contradictions":[("minimum wages reduce unionization incentive","minimum wages complement union organizing")],"gt_conf":"low"},
    {"q":"Does minimum wage timing and phase-in affect employment outcomes?",
     "gt_views":["gradual phase-in minimum wages reduces disemployment relative immediate increase","phase-in allows firm adjustment capital substitution investment smoothing","phase-in effects statistically indistinguishable from immediate implementation outcomes"],
     "gt_contradictions":[("phase-in reduces disemployment adjustment","phase-in effects indistinguishable from immediate")],"gt_conf":"low"},
    {"q":"What is the evidence on minimum wage and business productivity?",
     "gt_views":["minimum wages increase productivity through worker effort efficiency wages","no significant productivity gains from minimum wages offset labor cost","Draca Machin Van Reenen find no productivity increase UK minimum wage"],
     "gt_contradictions":[("minimum wages increase productivity effort wages","no significant productivity gains offset cost")],"gt_conf":"medium"},
    {"q":"Do minimum wages affect part-time versus full-time employment differently?",
     "gt_views":["employers shift full-time to part-time employment to avoid minimum wage threshold","part-time employment increase minimum wage reduces hours per worker","full-time part-time substitution minimal most minimum wage workers already part-time"],
     "gt_contradictions":[("employers shift full-time to part-time","full-time part-time substitution minimal")],"gt_conf":"medium"},
    {"q":"What is the evidence for minimum wages reducing wage theft?",
     "gt_views":["minimum wage increases reduce wage theft employer compliance improves","wage theft worst at lowest wages minimum wage reduces compliance gap","enforcement not minimum wage level determines wage theft prevalence"],
     "gt_contradictions":[("minimum wage reduces wage theft compliance","enforcement determines wage theft not level")],"gt_conf":"low"},
    {"q":"Do minimum wages have different effects in tight versus slack labor markets?",
     "gt_views":["tight labor markets minimum wages redundant wages already above minimum","slack labor markets minimum wages bite more employment effects larger","business cycle interaction minimum wage effects stronger recessions weaker booms"],
     "gt_contradictions":[("tight markets minimum wages redundant","slack markets employment effects larger minimum wage")],"gt_conf":"medium"},
    {"q":"What is the international evidence on minimum wages and economic growth?",
     "gt_views":["international evidence minimum wages do not reduce economic growth OECD","high minimum wages reduce business investment reduce long-run growth","minimum wage growth effect neutral dominated by other macroeconomic factors"],
     "gt_contradictions":[("minimum wages no growth reduction OECD","high minimum wages reduce investment long-run growth")],"gt_conf":"medium"},
    {"q":"Do minimum wages affect the underground or informal economy?",
     "gt_views":["minimum wages push low-wage workers informal economy tax avoidance","informal economy expansion minimum wage drives compliance avoidance","informal economy response minimum wages limited most workers formal economy"],
     "gt_contradictions":[("minimum wages push workers informal economy","informal economy response minimum wages limited")],"gt_conf":"low"},
  ],

  "climate": [
    {"q":"What is the equilibrium climate sensitivity to doubled CO2?",
     "gt_views":["ECS likely range 2.5-4.0 degrees Celsius best estimate 3 degrees IPCC AR6","Lewis Curry energy balance estimates ECS 1.7-2.5 degrees lower end","Sherwood 2020 multiple lines evidence ECS 2.6-3.9 degrees high confidence"],
     "gt_contradictions":[("ECS 3 degrees IPCC best estimate","ECS 1.7-2.5 lower energy balance"),("Sherwood high confidence 2.6-3.9","Lewis Curry lower estimate")],"gt_conf":"low"},
    {"q":"Can individual extreme weather events be attributed to climate change?",
     "gt_views":["attribution science probability ratio method attributes events climate change","extreme event attribution quantifies how much more likely events are with warming","attribution inherently probabilistic cannot attribute individual events definitively"],
     "gt_contradictions":[("attribution quantifies event probability change","cannot attribute individual events definitively")],"gt_conf":"low"},
    {"q":"How certain are scientists about human contribution to recent warming?",
     "gt_views":["IPCC AR6 extremely likely more than 95 percent human contribution since 1950","detection attribution studies attribute 100-110 percent warming to human activities","residual natural variability makes precise attribution range uncertain"],
     "gt_contradictions":[("95 percent confident human contribution","precise attribution range uncertain residual")],"gt_conf":"low"},
    {"q":"What is the transient climate response and how does it differ from ECS?",
     "gt_views":["TCR lower than ECS ocean heat uptake delay warming commitment","TCR approximately 1.8 degrees range 1.2-2.4 IPCC AR6","TCR more policy-relevant than ECS for near-term mitigation decisions"],
     "gt_contradictions":[("TCR lower ECS ocean delay","TCR range 1.2-2.4 substantial uncertainty")],"gt_conf":"low"},
    {"q":"Are aerosol forcings well understood for climate projections?",
     "gt_views":["aerosol forcing largest uncertainty anthropogenic forcing estimates IPCC","recent improved aerosol measurement satellite constrain forcing better","aerosol uncertainty propagates climate sensitivity uncertainty cannot separate"],
     "gt_contradictions":[("aerosol forcing large uncertainty","improved measurement better constrained aerosol")],"gt_conf":"medium"},
    {"q":"How reliable are climate models for regional temperature predictions?",
     "gt_views":["climate models reproduce global mean temperature well regional more uncertain","CMIP6 models improved regional precipitation temperature simulation skill","regional model projections show wide spread contradictory outcomes same region"],
     "gt_contradictions":[("climate models improved regional simulation","regional projections wide spread contradictory")],"gt_conf":"medium"},
    {"q":"Is there a possibility of crossing climate tipping points this century?",
     "gt_views":["tipping points Arctic sea ice Greenland West Antarctic possible 1.5-2 degrees","tipping element cascades unlikely without prolonged high forcing scenarios","tipping point risks highly uncertain low probability high consequence precautionary"],
     "gt_contradictions":[("tipping points possible 1.5-2 degrees warming","tipping cascades unlikely prolonged forcing required")],"gt_conf":"medium"},
    {"q":"How much sea level rise should we expect by 2100?",
     "gt_views":["IPCC AR6 likely range 0.3-1.0 meters SLR by 2100 scenario dependent","ice sheet instability could cause 2+ meters low probability high consequence","low emissions scenario limits SLR to 0.3-0.5 meters achievable"],
     "gt_contradictions":[("likely range 0.3-1.0 meters IPCC","ice sheet instability 2 plus meters possible")],"gt_conf":"medium"},
    {"q":"What role does solar variability play in recent climate change?",
     "gt_views":["solar forcing small compared anthropogenic since 1980 slight cooling trend solar","solar variability natural cycles explain some pre-industrial climate change","no credible evidence solar output explains post-1970 warming trend"],
     "gt_contradictions":[("solar forcing small compared anthropogenic","solar cycles explain some natural variability")],"gt_conf":"low"},
    {"q":"How do cloud feedbacks contribute to climate sensitivity uncertainty?",
     "gt_views":["low-cloud feedback largest uncertainty in climate sensitivity estimates","CMIP6 models show higher climate sensitivity due to stronger cloud feedbacks","observational constraints on cloud feedbacks tighten ECS range substantially Sherwood"],
     "gt_contradictions":[("cloud feedback largest uncertainty ECS","observational constraints tighten cloud feedback range")],"gt_conf":"low"},
    {"q":"Is the global surface temperature warming trend robust?",
     "gt_views":["multiple independent surface temperature datasets agree global warming trend","satellite tropospheric temperatures consistent surface warming adjusted"],
     "gt_contradictions":[("multiple independent datasets agree warming","satellite data initially showed discrepancy adjusted")],"gt_conf":"low"},
    {"q":"What does paleoclimate evidence imply about future warming?",
     "gt_views":["paleoclimate evidence supports ECS 2.5-4 degrees Celsius from glacial interglacial","CO2 paleoclimate relationship stable supports current projections","paleoclimate forcings differ from modern CO2 forcing interpretation uncertain"],
     "gt_contradictions":[("paleoclimate supports ECS range current projections","paleoclimate forcings different modern uncertain")],"gt_conf":"low"},
    {"q":"Are methane feedbacks from permafrost captured in climate projections?",
     "gt_views":["permafrost thaw methane release not fully represented current Earth system models","IPCC projections underestimate warming permafrost feedback omission","permafrost carbon feedback adds 0.1-0.5 degrees additional warming by 2100"],
     "gt_contradictions":[("permafrost feedback not fully represented underestimate","permafrost adds modest 0.1-0.5 additional warming")],"gt_conf":"medium"},
    {"q":"How much warming is already committed from current greenhouse gas concentrations?",
     "gt_views":["committed warming approximately 0.3-0.5 degrees even zero emissions today","zero emissions commitment scenario shows further modest warming unavoidable","committed warming highly uncertain depends cloud feedback aerosol masking"],
     "gt_contradictions":[("committed warming 0.3-0.5 degrees unavoidable","committed warming highly uncertain aerosol")],"gt_conf":"medium"},
    {"q":"Does climate sensitivity vary across different warming periods?",
     "gt_views":["effective climate sensitivity may vary as feedbacks change with background climate","paleo evidence suggests ECS higher warm climates than cold climates","variability in apparent ECS reflects forcing uncertainty not true ECS change"],
     "gt_contradictions":[("ECS varies warm cold background climate","apparent ECS variability reflects forcing uncertainty not true change")],"gt_conf":"medium"},
    {"q":"Are climate model hot spots in the tropical upper troposphere observed?",
     "gt_views":["tropical tropospheric warming hotspot predicted models observed satellite data","early satellite datasets showed discrepancy tropical hotspot now resolved","tropical hotspot magnitude smaller observations than models systematic bias"],
     "gt_contradictions":[("tropical hotspot observed satellite reconciled","hotspot magnitude smaller observations systematic bias remains")],"gt_conf":"medium"},
    {"q":"Are extreme precipitation events increasing due to climate change?",
     "gt_views":["Clausius-Clapeyron 7 percent per degree increase extreme precipitation thermodynamic","observed increase heavy precipitation events consistent climate projections","changes in circulation dynamical component complicate attribution precipitation extremes"],
     "gt_contradictions":[("Clausius-Clapeyron thermodynamic increase observed","dynamical circulation complicates attribution precipitation")],"gt_conf":"low"},
    {"q":"How does internal climate variability affect 20-year temperature trends?",
     "gt_views":["internal variability can mask or amplify forced trend up to 0.2 degrees 20 years","ENSO Pacific Decadal Oscillation drive decadal variability pause acceleration","internal variability cannot explain post-1970 long-term warming trend"],
     "gt_contradictions":[("internal variability masks trend 0.2 degrees","internal variability cannot explain long-term trend")],"gt_conf":"low"},
    {"q":"Does satellite temperature data agree with surface station records?",
     "gt_views":["satellite lower troposphere data now largely agrees surface records corrected","RSS UAH satellite products show similar warming trends surface datasets","satellite and surface records agreement improved after corrections orbital drift"],
     "gt_contradictions":[("satellite records now agree surface corrected","satellite surface records differ methodology dependent")],"gt_conf":"low"},
    {"q":"Are current emission reduction pledges sufficient to meet Paris targets?",
     "gt_views":["current NDC pledges insufficient 2.7-3 degrees warming 21st century","pledges strengthened 2030 still close 2 degree Paris goal narrowly","net zero 2050 pledges if implemented consistent 1.5-2 degree pathway"],
     "gt_contradictions":[("pledges insufficient 2.7-3 degrees warming","net zero pledges consistent 1.5-2 degrees pathway")],"gt_conf":"medium"},
    {"q":"What is the scientific evidence on low-end climate sensitivity below 2 degrees?",
     "gt_views":["low ECS below 1.5 degrees inconsistent multiple lines evidence","energy balance studies support low ECS 1.5-2 degrees plausible tail","AR6 assessed very unlikely ECS below 2 degrees given paleoclimate constraints"],
     "gt_contradictions":[("low ECS below 1.5 inconsistent evidence","energy balance supports low ECS 1.5-2 plausible")],"gt_conf":"low"},
    {"q":"How has attribution science changed our understanding of extreme events?",
     "gt_views":["attribution studies show climate change made specific events more probable 2-20x","probabilistic attribution provides quantitative risk communication for policy","attribution results strongly model-dependent different models give different conclusions"],
     "gt_contradictions":[("attribution quantifies climate change probability increase","attribution results model-dependent different conclusions")],"gt_conf":"low"},
    {"q":"Is there evidence of acceleration of glacier and ice sheet mass loss?",
     "gt_views":["GRACE satellite data shows acceleration Greenland West Antarctic mass loss","glacier mass loss globally accelerating since 1990s ice loss doubling","some glaciers showing temporary mass gain due to increased snowfall interior"],
     "gt_contradictions":[("ice mass loss accelerating globally","some glaciers temporary mass gain interior")],"gt_conf":"low"},
    {"q":"Does increased CO2 stimulate plant growth enough to offset warming?",
     "gt_views":["CO2 fertilization effect increases plant productivity globally greening Earth","fertilization effect limited by nutrient water availability not unlimited CO2","any CO2 plant benefit swamped by warming drought extreme heat crop effects"],
     "gt_contradictions":[("CO2 fertilization greening Earth benefit","fertilization limited nutrients not offsetting warming")],"gt_conf":"medium"},
    {"q":"How do economic damages from climate change scale with temperature increase?",
     "gt_views":["economic damages nonlinear above 2 degrees catastrophic at 4 degrees","damage functions highly uncertain calibrated limited historical data","economic damages at 2-3 degrees manageable 1-3 percent GDP meta-analysis"],
     "gt_contradictions":[("damages nonlinear catastrophic above 2 degrees","damages manageable 1-3 percent GDP 2-3 degrees")],"gt_conf":"medium"},
    {"q":"Is the rate of ocean acidification consistent with CO2 forcing models?",
     "gt_views":["ocean pH decline 0.1 units since 1900 consistent CO2 forcing projections","ocean acidification rate at upper end model projections coastal measurements","biological impacts acidification greater than pH alone suggests threshold effects"],
     "gt_contradictions":[("ocean acidification consistent projections","acidification biological impacts greater than pH models suggest")],"gt_conf":"low"},
    {"q":"What is the confidence level in climate projections beyond 2100?",
     "gt_views":["multi-century projections highly uncertain internal variability emission scenarios","physical understanding of long-term warming robust even with emission uncertainty","2200-2300 projections so scenario dependent as to be minimally informative"],
     "gt_contradictions":[("physical understanding robust long-term","multi-century projections highly uncertain minimally informative")],"gt_conf":"medium"},
    {"q":"Does urban heat island bias affect the global temperature record?",
     "gt_views":["urban heat island effect removed through station selection adjustment methods","urban heat island bias small 0.002-0.01 degrees per decade in adjusted records","contamination of global temperature record by urbanization cannot be fully removed"],
     "gt_contradictions":[("urban bias removed adjusted records small","contamination cannot be fully removed uncertain")],"gt_conf":"low"},
    {"q":"What role does water vapor feedback play in amplifying warming?",
     "gt_views":["water vapor feedback nearly doubles initial CO2 forcing well established","water vapor feedback constrained by observations models agree within 10 percent","water vapor feedback depends on regional atmospheric circulation difficult to isolate"],
     "gt_contradictions":[("water vapor feedback doubles warming well established","water vapor feedback difficult isolate regional circulation")],"gt_conf":"low"},
    {"q":"What fraction of recent warming is attributable to land use change versus CO2?",
     "gt_views":["land use change contributes 30-50 percent additional regional warming urban areas","globally land use change contributes small fraction total anthropogenic warming","land use change forcing poorly constrained large uncertainty IPCC assessment"],
     "gt_contradictions":[("land use regional warming 30-50 percent","globally land use small fraction poorly constrained")],"gt_conf":"medium"},
    {"q":"Are climate sensitivity estimates converging between energy balance and CMIP models?",
     "gt_views":["Sherwood 2020 assessment narrows ECS range 2.5-4.0 reconciling methods","energy balance lower estimates ECS CMIP higher estimates still diverge","apparent convergence reflects shared assumptions not independent constraints"],
     "gt_contradictions":[("Sherwood narrows ECS range reconciling","energy balance CMIP still diverge independent estimates")],"gt_conf":"medium"},
    {"q":"What would a low-end climate sensitivity scenario mean for policy?",
     "gt_views":["low ECS 1.5-2 degrees reduces urgency still requires significant decarbonization","low ECS does not eliminate impacts adaptation requirements still substantial","precautionary principle applies even if ECS lower risk of high ECS irreversible"],
     "gt_contradictions":[("low ECS reduces urgency decarbonization","precautionary principle applies even low ECS irreversible")],"gt_conf":"low"},
    {"q":"How should deep uncertainty in climate projections affect policy decisions?",
     "gt_views":["deep uncertainty supports precautionary mitigation avoid irreversible outcomes","under uncertainty risk-cost analysis favors immediate significant mitigation","uncertainty about damages argues for waiting for better information adaptive approaches"],
     "gt_contradictions":[("uncertainty supports precautionary immediate mitigation","uncertainty argues waiting better information adaptive")],"gt_conf":"medium"},
    {"q":"Does geoengineering stratospheric aerosol injection pose unacceptable risks?",
     "gt_views":["stratospheric aerosol injection rapid termination shock worse than gradual warming","SAI governance risks geopolitical conflict unilateral deployment","SAI may reduce warming risks while causing regional precipitation disruption"],
     "gt_contradictions":[("SAI termination shock unacceptable","SAI reduces warming risks precipitation trade-off")],"gt_conf":"medium"},
    {"q":"What is the evidence for acceleration of heat extreme events?",
     "gt_views":["heat extremes increasing faster than mean temperature shift variance increase","global warming shifted distribution heat extremes now 4-5 standard deviations","heat extreme acceleration consistent multiple attribution studies models observations"],
     "gt_contradictions":[("heat extremes faster mean temperature variance shift","heat extremes consistent standard shift distribution")],"gt_conf":"low"},
    {"q":"How does climate change affect Atlantic hurricane frequency and intensity?",
     "gt_views":["hurricane intensity increasing proportion category 4-5 storms climate change","global hurricane frequency may decrease total count but intensity increases","Atlantic hurricane activity driven partly by natural variability AMO conflating"],
     "gt_contradictions":[("hurricane intensity increasing climate change","total frequency decrease intensity increase"),("AMO natural variability conflating attribution","clear climate change signal hurricanes")],"gt_conf":"medium"},
    {"q":"Is it too late to avoid 2 degrees Celsius warming?",
     "gt_views":["remaining carbon budget 2 degrees allows modest emissions decades","net zero 2050 achievable pathway consistent below 2 degrees warming","committed warming cumulative emissions make 1.5 degrees nearly unavoidable 2 degrees difficult"],
     "gt_contradictions":[("net zero 2050 consistent below 2 degrees","1.5 nearly unavoidable 2 degrees difficult")],"gt_conf":"medium"},
    {"q":"What does the IPCC review process add to climate science confidence?",
     "gt_views":["IPCC synthesis provides authoritative conservative consensus across thousands studies","IPCC assessment process rigorous peer review adds substantial confidence","IPCC consensus process delays cutting-edge findings produces conservative underestimates"],
     "gt_contradictions":[("IPCC rigorous review adds confidence","IPCC conservative process delays findings underestimates")],"gt_conf":"low"},
    {"q":"Are carbon cycle feedbacks adequately represented in earth system models?",
     "gt_views":["carbon cycle feedbacks permafrost soil respiration not fully represented ESMs","CMIP6 ESMs show large spread carbon cycle feedback strength uncertain","carbon cycle feedback uncertainty adds 0.3-1.0 degrees additional warming by 2100"],
     "gt_contradictions":[("carbon cycle not fully represented ESMs","carbon cycle feedback uncertainty quantified CMIP6")],"gt_conf":"medium"},
    {"q":"What does the Roe and Baker argument say about climate sensitivity uncertainty?",
     "gt_views":["Roe Baker fat tail in ECS distribution from summing near-unity feedbacks structural","ECS fat tail uncertainty implies precautionary case strong for high ECS","fat tail argument criticized for ignoring physical constraints that narrow ECS"],
     "gt_contradictions":[("fat tail structural implies precautionary high ECS","fat tail criticized ignored physical constraints narrow ECS")],"gt_conf":"medium"},
    {"q":"What role do clouds play in climate projections uncertainty?",
     "gt_views":["low-cloud tropical feedback sign positive uncertain magnitude primary ECS uncertainty","high cloud feedback better constrained positive warming tropics","recent observational constraint cloud feedbacks narrows spread climate projections"],
     "gt_contradictions":[("low-cloud feedback uncertain primary ECS uncertainty","observational constraints narrow cloud feedback spread")],"gt_conf":"low"},
    {"q":"How does climate change interact with air quality and health outcomes?",
     "gt_views":["higher temperatures increase ozone wildfire smoke health costs significantly","CO2 emission reduction provides co-benefits air quality health globally","climate health impacts unevenly distributed poor populations highest exposure"],
     "gt_contradictions":[("climate change increases air quality health costs","CO2 reduction co-benefits air quality health")],"gt_conf":"low"},
    {"q":"What is the consensus on droughts increasing due to climate change?",
     "gt_views":["meteorological droughts increase projected thermodynamic soil moisture loss","Palmer Drought Severity Index shows increase soil moisture drought regions","soil moisture and runoff measures disagree making drought attribution contested"],
     "gt_contradictions":[("meteorological droughts increase thermodynamic","soil moisture runoff measures disagree attribution contested")],"gt_conf":"medium"},
    {"q":"What is the evidence on climate change and wildfire frequency?",
     "gt_views":["climate change increases wildfire area burned temperature drought drying vegetation","wildfire increase partly driven by land management fuel accumulation not only climate","attribution wildfire climate change strong western North America Mediterranean"],
     "gt_contradictions":[("climate change increases wildfire area burned","land management fuel accumulation partly responsible")],"gt_conf":"low"},
    {"q":"How does climate change affect biodiversity and species extinction risk?",
     "gt_views":["climate change threatens 30 percent species extinction risk 2 degrees warming","species range shifts adaptation may offset some extinction risk","extinction risk estimates highly model-dependent uncertain magnitude"],
     "gt_contradictions":[("climate change threatens species extinction 30 percent","adaptation range shifts offset extinction risk")],"gt_conf":"medium"},
    {"q":"What is the confidence in global mean temperature records?",
     "gt_views":["multiple independent temperature dataset groups agree global mean warming trend","HadCRUT Berkeley NOAA NASA datasets closely agree global temperature","coverage bias interpolation affect uncertainty bounds temperature records"],
     "gt_contradictions":[("multiple datasets agree global warming","coverage bias uncertainty affects records")],"gt_conf":"low"},
    {"q":"How do climate models compare to observed warming since 1970?",
     "gt_views":["CMIP model ensemble mean closely tracks observed warming since 1970","some CMIP6 models run too hot overestimate observed warming","observational uncertainty overlaps with model spread confirms model skill"],
     "gt_contradictions":[("CMIP models track observed warming closely","some CMIP6 models overestimate warming")],"gt_conf":"low"},
    {"q":"Is there scientific consensus on climate change?",
     "gt_views":["97 percent scientific consensus human-caused warming Cook 2013 Oreskes 2004","consensus level overstated by studies counting abstracts not full papers","consensus exists on warming but not on magnitude policy response uncertainty larger"],
     "gt_contradictions":[("97 percent consensus established","consensus level overstated methodology problems")],"gt_conf":"low"},
    {"q":"What is the relationship between climate sensitivity and extreme weather?",
     "gt_views":["higher climate sensitivity implies more frequent intense extremes nonlinear","extreme weather impacts scale with global mean temperature sensitivity matters","extreme weather attribution independent of ECS uncertainty near-term projections robust"],
     "gt_contradictions":[("higher sensitivity more extremes nonlinear","extreme attribution independent ECS uncertainty near-term")],"gt_conf":"medium"},
    {"q":"How well do climate models simulate decadal variability?",
     "gt_views":["climate models reproduce observed decadal variability patterns ENSO PDO","internal variability chaotic unpredictable models cannot replicate specific sequences","decadal hindcast skill limited but forced response signal extractable from ensemble"],
     "gt_contradictions":[("models reproduce decadal variability patterns","internal variability unpredictable specific sequences")],"gt_conf":"medium"},
  ],

  "diet_cvd": [
    {"q":"Does eating saturated fat increase the risk of coronary heart disease?",
     "gt_views":["saturated fat raises LDL cholesterol increases coronary heart disease risk meta-analysis","Siri-Tarino Chowdhury meta-analyses find no significant saturated fat CVD association","replacement nutrient matters replacing saturated fat refined carbohydrate increases risk"],
     "gt_contradictions":[("saturated fat increases CVD risk LDL","no significant saturated fat CVD association meta-analysis"),("replacement nutrient matters refined carb increases risk","saturated fat reduction beneficial regardless replacement")],"gt_conf":"high"},
    {"q":"Should dietary guidelines recommend limiting saturated fat intake?",
     "gt_views":["dietary guidelines recommend saturated fat below 10 percent energy cardiovascular benefit","dietary guidelines saturated fat recommendation not well supported RCT evidence","guideline recommendations should specify replacement nutrient not just reduce saturated fat"],
     "gt_contradictions":[("dietary guidelines recommend limit saturated fat","guidelines not well supported RCT evidence"),("specify replacement nutrient","simply limit saturated fat sufficient")],"gt_conf":"high"},
    {"q":"Does replacing saturated fat with polyunsaturated fat reduce cardiovascular risk?",
     "gt_views":["replacing saturated fat polyunsaturated fat reduces coronary events meta-analysis 10 percent","Jakobsen specific PUFA replacement reduces CHD risk linoleic acid benefit","not all PUFAs equal omega-3 more beneficial omega-6 inflammatory"],
     "gt_contradictions":[("PUFA replacement reduces CHD risk","not all PUFAs equal omega-6 inflammatory"),("Jakobsen linoleic acid beneficial","omega-6 may increase inflammation concern")],"gt_conf":"medium"},
    {"q":"What is the relationship between dietary cholesterol and blood cholesterol?",
     "gt_views":["dietary cholesterol has modest effect blood cholesterol at physiological intakes","eggs and shellfish dietary cholesterol raise LDL small amount modest effect","dietary cholesterol less important than saturated fat for LDL determination"],
     "gt_contradictions":[("dietary cholesterol modest effect blood cholesterol","dietary cholesterol important CVD risk factor eggs")],"gt_conf":"medium"},
    {"q":"Does the Mediterranean diet reduce cardiovascular mortality?",
     "gt_views":["PREDIMED trial Mediterranean diet reduced major cardiovascular events 30 percent","Mediterranean diet benefits extend beyond fat type dietary pattern synergy","PREDIMED retraction reanalysis effect size reduced but benefit maintained"],
     "gt_contradictions":[("PREDIMED Mediterranean diet 30 percent cardiovascular reduction","PREDIMED retraction reduced effect size maintained")],"gt_conf":"medium"},
    {"q":"Is the low-fat dietary paradigm supported by randomized controlled trial evidence?",
     "gt_views":["Women's Health Initiative low-fat intervention no significant CVD reduction 8 years","low-fat paradigm lacks RCT support total fat less important than fat quality","low-fat diet reduces cholesterol and LDL when properly implemented benefit real"],
     "gt_contradictions":[("WHI low-fat no significant CVD reduction","low-fat reduces cholesterol benefit real")],"gt_conf":"high"},
    {"q":"Does total dietary fat intake predict cardiovascular disease risk?",
     "gt_views":["total fat intake not significantly associated CVD risk dietary pattern important","PURE study high fat intake associated lower mortality compared high carbohydrate","total fat weak predictor CVD fat type and food source more important"],
     "gt_contradictions":[("total fat not significantly associated CVD","PURE high fat lower mortality higher carbohydrate"),("fat type more important than total","PURE favors higher fat intake")],"gt_conf":"high"},
    {"q":"Are all saturated fatty acids equally harmful for cardiovascular health?",
     "gt_views":["different saturated fatty acids vary LDL raising effects stearic acid neutral","palmitic acid most common saturated fat significantly raises LDL cardiovascular risk","food matrix modifies saturated fatty acid effects dairy different from meat"],
     "gt_contradictions":[("stearic acid neutral cardiovascular","palmitic acid raises LDL risk"),("food matrix modifies effects dairy different","saturated fatty acid type determines risk")],"gt_conf":"medium"},
    {"q":"What is the effect of red meat consumption on cardiovascular disease?",
     "gt_views":["processed meat increases CVD mortality more than unprocessed red meat","unprocessed red meat no significant independent CVD risk association meta-analysis","red meat heme iron nitrates mechanisms CVD beyond saturated fat content"],
     "gt_contradictions":[("processed meat more harmful than unprocessed","unprocessed red meat no significant CVD risk"),("additional mechanisms heme iron","simply saturated fat content explains")],"gt_conf":"medium"},
    {"q":"Is LDL cholesterol the correct target for dietary interventions?",
     "gt_views":["LDL-C primary causal CVD risk factor Mendelian randomization genetic evidence","small dense LDL particle size better CVD predictor than total LDL-C","ApoB better predictor CVD than LDL-C dietary intervention target"],
     "gt_contradictions":[("LDL primary causal target dietary","ApoB small dense LDL better predictor than LDL")],"gt_conf":"medium"},
    {"q":"What does the Women's Health Initiative dietary fat trial show?",
     "gt_views":["WHI low-fat diet trial no significant reduction CVD events over 8 years","WHI limited by poor low-fat diet adherence does not exclude diet-CVD link","WHI low-fat group achieved modest fat reduction insufficient test strong hypothesis"],
     "gt_contradictions":[("WHI no significant CVD reduction","WHI poor adherence insufficient test of hypothesis")],"gt_conf":"high"},
    {"q":"Do low-carbohydrate diets improve cardiovascular risk factors more than low-fat?",
     "gt_views":["low-carbohydrate diets improve HDL triglycerides better than low-fat short-term","low-fat diets reduce LDL better than low-carbohydrate long-term CVD risk","dietary pattern adherence most important factor not macronutrient ratio"],
     "gt_contradictions":[("low-carb improves HDL triglycerides short-term","low-fat reduces LDL better long-term")],"gt_conf":"medium"},
    {"q":"What is the food matrix effect on saturated fat cardiovascular risk?",
     "gt_views":["dairy fermented cheese yogurt does not increase CVD despite saturated fat","food matrix modulates absorption metabolism of saturated fatty acids","saturated fat in whole food context different from isolated saturated fat supplementation"],
     "gt_contradictions":[("dairy cheese no CVD increase despite saturated fat","saturated fat regardless source increases CVD risk")],"gt_conf":"medium"},
    {"q":"Does butter versus olive oil affect cardiovascular disease risk differently?",
     "gt_views":["olive oil substitution for butter reduces CVD risk 7 percent per 10g replacement","butter raises LDL more than olive oil cardiovascular advantage clear","butter versus olive oil difference modest overall dietary pattern dominates"],
     "gt_contradictions":[("olive oil substitution reduces CVD butter","butter versus olive oil difference modest pattern dominates")],"gt_conf":"medium"},
    {"q":"What does the PREDIMED trial evidence show about Mediterranean diet and CVD?",
     "gt_views":["PREDIMED Mediterranean diet olive oil nuts reduced major events 30 percent","PREDIMED retraction randomization concerns effect size reanalysis smaller","Mediterranean diet benefit confirmed multiple cohort observational studies beyond PREDIMED"],
     "gt_contradictions":[("PREDIMED 30 percent reduction strong evidence","PREDIMED retraction concerns smaller effect reanalysis")],"gt_conf":"medium"},
    {"q":"Are eggs harmful for cardiovascular health due to dietary cholesterol?",
     "gt_views":["eggs dietary cholesterol modest effect blood cholesterol moderate consumption safe","JACC study egg consumption associated higher CVD mortality dose-dependent","dietary cholesterol guidelines removed 2015 insufficient RCT evidence for restriction"],
     "gt_contradictions":[("eggs safe moderate consumption modest effect","egg consumption CVD mortality dose-dependent JACC")],"gt_conf":"medium"},
    {"q":"What is the evidence for omega-3 fatty acid supplementation and CVD prevention?",
     "gt_views":["high-dose omega-3 EPA REDUCE-IT trial reduced cardiovascular events 25 percent","ASCEND ORIGIN trials omega-3 supplementation no significant CVD benefit","mineral oil placebo REDUCE-IT inflated apparent omega-3 benefit artifact"],
     "gt_contradictions":[("REDUCE-IT high-dose omega-3 25 percent reduction","ASCEND ORIGIN no significant benefit"),("REDUCE-IT valid result","mineral oil placebo artifact inflated benefit")],"gt_conf":"high"},
    {"q":"Are vegetable oils high in omega-6 fats beneficial or harmful for the heart?",
     "gt_views":["omega-6 linoleic acid reduces LDL CVD risk beneficial replacing saturated fat","high omega-6 intake may increase inflammation arachidonic acid concern","observational evidence omega-6 oils beneficial cardiovascular outcomes consistent"],
     "gt_contradictions":[("omega-6 linoleic acid reduces CVD beneficial","omega-6 inflammation arachidonic acid concern")],"gt_conf":"medium"},
    {"q":"What does the Seven Countries Study reanalysis tell us about saturated fat?",
     "gt_views":["Seven Countries Study established dietary saturated fat CHD association foundational","Keys cherry-picked countries selective data distorted saturated fat conclusion","modern reanalysis Seven Countries confirms saturated fat CHD association robust"],
     "gt_contradictions":[("Seven Countries Study foundational robust association","Keys cherry-picked selective distortion")],"gt_conf":"high"},
    {"q":"Does the replacement nutrient matter when reducing saturated fat?",
     "gt_views":["replacing saturated fat refined carbohydrate increases CVD risk not beneficial","replacing saturated fat unsaturated fat reduces CVD risk benefit real","replacement by protein shows modest CVD benefit nutrient substitution important"],
     "gt_contradictions":[("replacement refined carb increases risk","replacement unsaturated fat reduces risk")],"gt_conf":"medium"},
    {"q":"Is there a J-shaped relationship between fat intake and cardiovascular outcomes?",
     "gt_views":["PURE study suggests J-shaped mortality fat intake lowest quartile higher mortality","J-shaped relationship confounded malnutrition low-income countries low fat","linear or threshold relationship fat CVD supported higher-income population studies"],
     "gt_contradictions":[("PURE J-shaped fat intake mortality","J-shaped confounded malnutrition low-income")],"gt_conf":"high"},
    {"q":"Do dairy saturated fats have the same cardiovascular effects as meat saturated fats?",
     "gt_views":["dairy whole fat cheese yogurt neutral or beneficial cardiovascular outcomes","meat saturated fat different cardiovascular effect than dairy fat food matrix","dairy and meat saturated fats chemically similar comparable cardiovascular effect"],
     "gt_contradictions":[("dairy fat neutral beneficial cardiovascular","dairy meat saturated fat chemically similar comparable effect")],"gt_conf":"medium"},
    {"q":"What does the PURE study evidence show about fat and cardiovascular outcomes?",
     "gt_views":["PURE study high fat intake lower mortality higher carbohydrate higher mortality","PURE findings confounded by food insecurity poverty low carb populations","PURE results support dietary pattern approach not universal fat restriction"],
     "gt_contradictions":[("PURE high fat lower mortality supports higher intake","PURE confounded poverty food insecurity not causal")],"gt_conf":"high"},
    {"q":"Are current cardiovascular dietary guidelines evidence-based and effective?",
     "gt_views":["dietary guidelines rigorous evidence review process systematic basis for recommendations","dietary guidelines insufficient RCT evidence most recommendations expert consensus","guideline implementation weak population adherence limits effectiveness evaluation"],
     "gt_contradictions":[("dietary guidelines rigorous evidence-based","guidelines insufficient RCT consensus-based")],"gt_conf":"high"},
    {"q":"Is coconut oil beneficial, neutral, or harmful for cardiovascular health?",
     "gt_views":["coconut oil raises LDL more than unsaturated oils cardiovascular concern","coconut oil raises HDL cholesterol simultaneously net effect uncertain","AHA advises against coconut oil cardiovascular risk based LDL evidence"],
     "gt_contradictions":[("coconut oil raises LDL harmful","coconut oil raises HDL uncertain net effect")],"gt_conf":"medium"},
    {"q":"Does dietary fat influence gut microbiome composition?",
     "gt_views":["high saturated fat diet shifts gut microbiome toward inflammation increased permeability","microbiome response dietary fat highly individual limits population-level guidance","gut microbiome dietary fat interaction mechanism CVD risk not clinically established"],
     "gt_contradictions":[("high saturated fat shifts microbiome inflammation","microbiome-fat interaction not clinically established CVD")],"gt_conf":"low"},
    {"q":"Are ketogenic diets beneficial or harmful for long-term cardiovascular health?",
     "gt_views":["ketogenic diets reduce triglycerides raise HDL improve short-term cardiometabolic markers","ketogenic diets raise LDL significantly concerning long-term cardiovascular risk","ketogenic diet long-term cardiovascular outcomes data insufficient RCT evidence"],
     "gt_contradictions":[("ketogenic improves cardiometabolic markers short-term","ketogenic raises LDL long-term concern"),("insufficient long-term RCT data","cardiometabolic improvement short-term")],"gt_conf":"medium"},
    {"q":"Do Mendelian randomization studies support the LDL-dietary fat relationship?",
     "gt_views":["Mendelian randomization confirms LDL causal CVD risk supports dietary LDL reduction","MR studies cannot test dietary interventions only genetic proxies limitation","genetic LDL studies support LDL causal role but not specific dietary approaches"],
     "gt_contradictions":[("MR confirms LDL causal CVD supports dietary","MR cannot test dietary interventions only proxies")],"gt_conf":"medium"},
    {"q":"What dietary pattern best reduces CVD risk in established heart disease?",
     "gt_views":["Mediterranean diet best evidence secondary prevention CVD Lyon Diet Heart Study","cardiac diet low saturated fat high fiber evidence from multiple trials","plant-based whole food dietary pattern reduces recurrent events strongest evidence"],
     "gt_contradictions":[("Mediterranean diet best secondary prevention evidence","plant-based whole food diet strongest evidence")],"gt_conf":"medium"},
    {"q":"Does dietary fat affect HDL cholesterol in cardiovascular-protective ways?",
     "gt_views":["saturated fat raises HDL cholesterol but this may not protect cardiovascular","HDL raising by saturated fat different from HDL raising by exercise lifestyle","HDL functionality not just concentration determines cardiovascular protective role"],
     "gt_contradictions":[("saturated fat raises HDL protective","saturated fat HDL raising different from lifestyle HDL")],"gt_conf":"medium"},
    {"q":"Is the association between saturated fat and CVD confounded by carbohydrate intake?",
     "gt_views":["confounding by carbohydrate intake distorts saturated fat CVD association key issue","isocaloric substitution studies address confounding show saturated fat harmful","carbohydrate quality not quantity confounds saturated fat association"],
     "gt_contradictions":[("carbohydrate confounding distorts saturated fat association","isocaloric substitution addresses confounding harm confirmed")],"gt_conf":"high"},
    {"q":"Do dietary interventions match drug interventions for CVD prevention efficacy?",
     "gt_views":["Mediterranean diet PREDIMED reduces CVD comparable or better than drug NNT","dietary interventions prevent CVD but with smaller absolute risk reductions than statins","combination diet and statins provides greater CVD protection than either alone"],
     "gt_contradictions":[("dietary intervention comparable drug NNT PREDIMED","dietary smaller absolute risk reduction than statins")],"gt_conf":"medium"},
    {"q":"What is the evidence on whole-fat versus low-fat dairy and CVD?",
     "gt_views":["whole-fat dairy not associated increased CVD risk observational meta-analysis","full-fat dairy raises LDL mechanistic cardiovascular concern","low-fat dairy may increase sugar refined carbohydrate intake replacing fat adverse"],
     "gt_contradictions":[("whole-fat dairy not associated CVD risk","full-fat dairy raises LDL cardiovascular concern")],"gt_conf":"medium"},
    {"q":"What are the cardiovascular effects of dietary fat in diabetic populations?",
     "gt_views":["type 2 diabetes low-carbohydrate diet superior glycemic control cardiovascular markers","saturated fat worse cardiovascular risk diabetic insulin-resistant populations","Mediterranean diet superior cardiovascular benefit diabetic populations PREDIMED-PLUS"],
     "gt_contradictions":[("low-carbohydrate superior glycemic CVD markers diabetic","saturated fat worse diabetic populations")],"gt_conf":"medium"},
    {"q":"What would the optimal dietary fat intake recommendation be based on current evidence?",
     "gt_views":["optimal dietary fat recommendation emphasize quality unsaturated over saturated","no optimal total fat percentage justified by evidence dietary pattern more important","optimal fat intake highly individual based on genetics metabolic phenotype context"],
     "gt_contradictions":[("quality unsaturated over saturated optimal recommendation","no optimal percentage pattern more important"),("pattern more important than total","individual genetics optimal")],"gt_conf":"high"},
    {"q":"Does replacing saturated fat with refined carbohydrates increase cardiovascular risk?",
     "gt_views":["replacing saturated fat refined carbohydrate increases CVD risk clear evidence","replacing saturated fat whole grains legumes reduces CVD risk substantially","low-fat diet advice led to refined carbohydrate increase contributing obesity epidemic"],
     "gt_contradictions":[("refined carb replacement increases CVD risk","whole grain replacement reduces CVD risk substantially")],"gt_conf":"high"},
    {"q":"What does the Astrup 2020 reassessment show about saturated fats and health?",
     "gt_views":["Astrup 2020 argues dairy saturated fat not harmful considers food matrix","Astrup reassessment controversial cherry-picks evidence food industry funding concern","food matrix principle supported multiple independent meta-analyses dairy neutral beneficial"],
     "gt_contradictions":[("Astrup dairy neutral food matrix principle","Astrup controversial funding concern cherry-picks")],"gt_conf":"high"},
    {"q":"Does high fat intake increase oxidative stress and endothelial dysfunction?",
     "gt_views":["high saturated fat meal acute endothelial dysfunction oxidative stress postprandially","chronic high fat diet endothelial dysfunction depends on fat quality not quantity","postprandial endothelial effects of fat not predictive chronic CVD outcomes"],
     "gt_contradictions":[("high saturated fat acute endothelial dysfunction","endothelial effects not predictive chronic CVD outcomes")],"gt_conf":"low"},
    {"q":"Is palm oil neutral or harmful for cardiovascular health?",
     "gt_views":["palm oil high palmitic acid raises LDL cardiovascular risk comparable butter","palm oil refined refined food processing may be worse than natural palm","palm oil replacing trans fats better cardiovascular outcome than hydrogenated oils"],
     "gt_contradictions":[("palm oil raises LDL harmful","palm oil better than hydrogenated trans fats")],"gt_conf":"medium"},
    {"q":"How does trans fat compare to saturated fat for cardiovascular risk?",
     "gt_views":["industrial trans fats clearly worse than saturated fat both LDL and HDL effects","trans fat elimination from food supply directly contributed CVD mortality reduction","ruminant trans fats different from industrial trans fats possibly neutral or beneficial"],
     "gt_contradictions":[("industrial trans fats clearly worse than saturated fat","ruminant trans fats possibly neutral beneficial")],"gt_conf":"medium"},
    {"q":"What is the evidence for monounsaturated fat and cardiovascular protection?",
     "gt_views":["monounsaturated fat oleic acid Mediterranean diet protective cardiovascular outcomes","MUFA cardiovascular benefit dependent on what it replaces saturated or carbohydrate","monounsaturated fat cardiovascular benefit modest compared polyunsaturated fat evidence"],
     "gt_contradictions":[("MUFA Mediterranean diet protective cardiovascular","MUFA benefit dependent replacement context modest compared PUFA")],"gt_conf":"medium"},
    {"q":"Does olive oil supplementation provide benefits beyond dietary pattern?",
     "gt_views":["olive oil extra virgin polyphenols provide cardiovascular benefit beyond fat composition","PREDIMED olive oil arm showed greatest benefit extra virgin quality matters","olive oil benefit cannot be separated from overall Mediterranean dietary pattern"],
     "gt_contradictions":[("olive oil polyphenols additional benefit beyond fat","olive oil benefit inseparable from Mediterranean pattern")],"gt_conf":"medium"},
    {"q":"Do dietary guidelines adequately account for individual genetic variation in fat metabolism?",
     "gt_views":["apoE genotype determines individual response dietary fat cholesterol recommendations","fat metabolism genetic variants show insufficient heterogeneity for population guidance","nutrigenomics field immature cannot yet individualize dietary fat recommendations effectively"],
     "gt_contradictions":[("apoE genotype determines individual fat response","nutrigenomics immature cannot individualize recommendations")],"gt_conf":"low"},
    {"q":"What evidence supports or refutes the dietary fat-cancer connection?",
     "gt_views":["high fat diet colorectal cancer risk through gut microbiome bile acid mechanisms","WHI low-fat diet no significant breast cancer reduction disappointing negative trial","dietary fat cancer connection weak confounded dietary pattern total calories"],
     "gt_contradictions":[("dietary fat cancer risk mechanisms gut microbiome","low-fat diet no significant cancer reduction WHI")],"gt_conf":"medium"},
    {"q":"Does dietary fat affect non-alcoholic fatty liver disease risk?",
     "gt_views":["high saturated fat diet promotes hepatic steatosis NAFLD progression","total caloric surplus not fat type drives NAFLD development","fructose excess more important than dietary fat in NAFLD pathogenesis"],
     "gt_contradictions":[("saturated fat promotes NAFLD steatosis","caloric surplus not fat type drives NAFLD")],"gt_conf":"medium"},
    {"q":"Is the Nordic diet as cardiovascular-protective as the Mediterranean diet?",
     "gt_views":["Nordic diet whole grains fish rapeseed oil comparable Mediterranean cardiovascular benefits","Mediterranean diet stronger evidence base RCT Nordic diet primarily observational","Nordic diet culturally appropriate northern populations equivalent cardiovascular protection"],
     "gt_contradictions":[("Nordic diet comparable Mediterranean cardiovascular","Mediterranean stronger RCT evidence Nordic observational")],"gt_conf":"low"},
    {"q":"Do omega-3 fatty acids reduce triglycerides significantly?",
     "gt_views":["high-dose omega-3 EPA DHA reduces triglycerides 20-30 percent well established","triglyceride reduction does not translate consistently cardiovascular event reduction","prescription omega-3 icosapent ethyl reduces CVD events REDUCE-IT different from supplements"],
     "gt_contradictions":[("omega-3 reduces triglycerides well established","triglyceride reduction inconsistent CVD event reduction")],"gt_conf":"medium"},
    {"q":"What is the role of dietary fat in blood pressure regulation?",
     "gt_views":["Mediterranean diet reduces blood pressure beyond salt reduction mechanisms","saturated fat raises blood pressure endothelial inflammation mechanism","dietary fat type not quantity modestly affects blood pressure confounded overall diet"],
     "gt_contradictions":[("Mediterranean diet reduces blood pressure","dietary fat blood pressure effect modest confounded overall diet")],"gt_conf":"low"},
    {"q":"Does the glycemic index interact with dietary fat for cardiovascular risk?",
     "gt_views":["low glycemic index diet combined with unsaturated fat reduces CVD risk synergistically","glycemic index dietary fat interaction not clinically meaningful independent effects","high glycemic diet amplifies saturated fat cardiovascular harm metabolic mechanism"],
     "gt_contradictions":[("glycemic index fat interaction synergistic","glycemic index fat effects independent not meaningful interaction")],"gt_conf":"low"},
    {"q":"What is the DASH diet evidence for cardiovascular risk reduction?",
     "gt_views":["DASH diet low saturated fat high fiber significantly reduces blood pressure CVD risk","DASH diet benefit primarily from sodium reduction not fat composition","DASH and Mediterranean diets both effective through partly overlapping mechanisms"],
     "gt_contradictions":[("DASH diet reduces CVD risk fat composition benefit","DASH benefit primarily sodium not fat")],"gt_conf":"medium"},
  ],
}

# ─── fallback corpus (15 papers per domain, 75 total) ─────────────────────────
FALLBACK_CORPUS = {
  "homework": [
    {"title":"Homework and Academic Achievement: A Meta-Analysis","year":2006,"authors":["Cooper, H.","Robinson, J.","Patall, E."],
     "abstract":"A meta-analysis of 60+ homework studies found positive correlations between homework and achievement for grades 7-12 (r=0.24) but near-zero correlations for elementary school (r=0.02). Effect sizes varied considerably by outcome measure. Studies using standardized tests showed smaller effects than those using teacher-assigned grades. The relationship was strongest for mathematics homework in secondary school. These findings suggest homework has conditional effects that depend critically on grade level, subject, and assignment design."},
    {"title":"Is More Homework Better? Evidence from an International Comparison","year":2015,"authors":["Fernandez-Alonso, R.","Suarez-Alvarez, J.","Muniz, J."],
     "abstract":"Using PISA data from 29 countries, we examined the relationship between homework and academic performance. Contrary to the assumption that more homework leads to better outcomes, we found diminishing returns above 90-100 minutes per day, with negative associations for very high homework loads. Cross-national variation was substantial: in some educational systems homework explained more variance in achievement than in others. Socioeconomic status moderated the homework-achievement relationship, with wealthier students more able to benefit from homework due to home resources."},
    {"title":"The Homework Debate: A Research Synthesis","year":2019,"authors":["Nunez, J.C.","Suarez, N.","Rosario, P."],
     "abstract":"We review three decades of homework research and identify key methodological limitations: most studies are correlational, use self-reported homework time, and fail to control for prior achievement. When high-quality experimental designs are used, homework effects are smaller than correlational studies suggest. Publication bias inflates positive findings. Assignment quality mediates effectiveness more than quantity. We call for a moratorium on assigning homework based solely on tradition, pending higher-quality evidence."},
    {"title":"Homework and Student Wellbeing: Stress, Sleep and Mental Health","year":2015,"authors":["Pope, D.","Miles, S.","Brown, M."],
     "abstract":"In a survey of 4,317 students at ten high-performing high schools, we found that 56% cited homework as a primary source of stress. Students averaging more than 3.1 hours of homework nightly experienced significantly elevated cortisol levels, disrupted sleep, and higher rates of anxiety symptoms. Academic achievement gains did not compensate for these wellbeing costs in high-homework conditions. We argue that existing homework policies prioritize achievement signals over student health at significant developmental cost."},
    {"title":"Homework and Equity: Differential Effects by Family Income","year":2018,"authors":["Kalenkoski, C.","Pabilonia, S."],
     "abstract":"Using time-use diary data, we demonstrate that homework amplifies socioeconomic achievement gaps rather than narrowing them. Students from low-income families spend equivalent time on homework as high-income peers but derive smaller achievement benefits, likely due to differences in parental educational capital, quiet study space, and access to reference materials. Homework effectiveness is conditioned on resource availability, making uniform homework policies distributionally regressive."},
    {"title":"Self-Regulation and Homework: Mediating Mechanisms","year":2017,"authors":["Ramdass, D.","Zimmerman, B."],
     "abstract":"Self-regulatory behaviors—goal setting, self-monitoring, self-evaluation—mediate the homework-achievement relationship. Students with higher self-efficacy complete more homework and benefit more from it. Homework can serve as a practice environment for developing self-regulation if assignments are appropriately scaffolded. However, poorly designed homework may undermine intrinsic motivation, particularly for younger students. Homework should be viewed as a self-regulation training opportunity, not merely content reinforcement."},
    {"title":"Rethinking Homework: Best Practices That Support Diverse Learners","year":2007,"authors":["Vatterott, C."],
     "abstract":"Traditional homework practices—uniform assignments, completion grades, punitive policies—are inconsistent with current understanding of learning science. Research-aligned alternatives include student choice in task type, practice-only homework, and eliminating homework grades. Districts that have adopted research-aligned homework policies report reduced family conflict, maintained or improved achievement, and teacher professional satisfaction. Eliminating homework in grades K-2 has no detectable negative achievement effects in follow-up studies."},
    {"title":"Homework Completion and Academic Achievement: A Multilevel Analysis","year":2012,"authors":["Nunez, J.C.","Suarez, N.","Cerezo, R."],
     "abstract":"Multilevel modeling of 1,015 elementary and secondary students revealed that homework completion rate, not assigned time, predicted achievement. A 10% increase in completion rate was associated with a 0.18 SD gain in mathematics achievement. Teacher homework quality (feedback, calibration to ability level) moderated this relationship. Students who received feedback on homework showed twice the achievement gain of those who had homework collected-only or not collected. These findings redirect attention from homework quantity to implementation quality."},
    {"title":"Homework and Academic Achievement: A Cross-National Survey","year":2013,"authors":["Trautwein, U.","Ludtke, O."],
     "abstract":"Across 50 countries participating in TIMSS, the homework-achievement association varied substantially. In East Asian countries, despite high homework loads, gains per hour were modest, suggesting diminishing returns. In Nordic countries with low homework loads, achievement was maintained through effective in-class instruction. Multilevel models showed that school-level homework practices explained more variance than individual student homework time, suggesting that schoolwide homework culture matters beyond individual compliance."},
    {"title":"Parents, Homework, and Academic Achievement: A Review","year":2010,"authors":["Patall, E.","Cooper, H.","Robinson, J."],
     "abstract":"A meta-analysis of 20 studies found that parental homework involvement had a small positive effect (d=0.09) on academic achievement in elementary school. Effects were more positive when parents provided emotional support versus directive assistance. Negative effects emerged when parents completed homework for children or provided incorrect help. Parental involvement was more beneficial for younger children and declined in importance as students developed autonomy. Recommendations include training parents in supportive (not directive) involvement strategies."},
    {"title":"The Effect of Homework on Student Learning: A Review","year":2014,"authors":["Cooper, H.","Valentine, J."],
     "abstract":"Reviewing homework effects across 20 years of research, we conclude that the evidence supports a causal role of homework on achievement, particularly for secondary students. Effect sizes from well-controlled experimental studies average d=0.20 for secondary and d=0.06 for elementary school. The type of homework matters: practice homework shows the strongest effects, preparation homework weaker effects, and extension homework negligible effects. The homework-achievement relationship is moderated by student motivation, ability, and home environment quality."},
    {"title":"Rethinking the Homework Experience: A Developmental Perspective","year":2016,"authors":["Silinskas, G.","Kikas, E."],
     "abstract":"A longitudinal study of 659 children from first to fourth grade showed that parental homework help became increasingly counterproductive as children aged. Direct assistance in grade 1 predicted higher achievement in grade 2, but direct assistance in grade 3 predicted lower achievement in grade 4. Autonomy-supportive parental behavior showed the reverse pattern. These findings suggest that homework policies should be developmental—structured parent-child collaboration in early years transitioning to student independence in later years."},
    {"title":"Homework and Student Stress: A Longitudinal Study","year":2021,"authors":["Galloway, M.","Conner, J.","Pope, D."],
     "abstract":"Following 4,213 high school students over three academic years, we documented that homework-related stress was the strongest predictor of academic burnout (beta=0.42), stronger than test stress or teacher expectations. Schools that reduced homework loads by 25% saw significant reductions in burnout without commensurate achievement decreases. Long-term outcomes at 18-month follow-up showed sustained wellbeing improvements. These findings support intentional homework reduction policies at the school level as a mental health intervention."},
    {"title":"Digital Homework Platforms and Academic Achievement","year":2022,"authors":["Roschelle, J.","Feng, M.","Murphy, R."],
     "abstract":"A randomized experiment with 2,850 middle school mathematics students compared adaptive digital homework to traditional paper homework. Digital homework students received immediate corrective feedback and adaptive difficulty adjustment. After one academic year, digital homework students outperformed controls by 0.14 standard deviations on standardized tests. Completion rates were 23% higher in the digital condition. The effect was larger for students with lower prior achievement, suggesting digital homework may reduce the achievement gap associated with home resource differences."},
    {"title":"Should Schools Eliminate Homework? Evidence from a Natural Experiment","year":2020,"authors":["Marion, S.","Buckley, J."],
     "abstract":"When a large urban school district eliminated homework in grades K-2 and restricted it to 30 minutes maximum in grades 3-5, we compared achievement trajectories across five cohorts using interrupted time series. Elementary achievement was unchanged across three academic years post-policy. Parent-reported family stress declined significantly. Teacher workload for grading decreased. The natural experiment provides the strongest existing evidence that homework elimination in elementary school produces no achievement harm while providing wellbeing benefits."},
  ],

  "statins": [
    {"title":"Primary Prevention with Statins: A Systematic Review and Meta-Analysis","year":2016,"authors":["Tonelli, M.","Lloyd, A.","Clement, F."],
     "abstract":"In a meta-analysis of 34 randomized controlled trials (n=182,000), statin therapy for primary prevention reduced major cardiovascular events by 25% (RR 0.75, 95% CI 0.70-0.81) and all-cause mortality by 10% (RR 0.90, 0.87-0.93). Benefits were present across all risk strata, including low-risk individuals (10-year risk <10%). Number needed to treat was 80 for 5 years to prevent one major event in average-risk individuals. These data support broader statin use in primary prevention than current guidelines recommend."},
    {"title":"Statins for Primary Prevention in Low-Risk Individuals: Harms May Outweigh Benefits","year":2019,"authors":["Byrne, P.","Cullinan, J.","Smith, S."],
     "abstract":"Re-analysis of primary prevention trial data reveals that absolute risk reductions from statins are modest. In individuals with 10-year cardiovascular risk below 7.5%, the number-needed-to-treat exceeds 200, while the number-needed-to-harm for diabetes onset is approximately 50. Myopathy risk is underestimated in trial data due to run-in period selection bias and under-reporting. We argue that for low-risk individuals, the risk-benefit profile of statin therapy is not clearly favorable, and guidelines should require shared decision-making with explicit absolute risk communication."},
    {"title":"Statin Myopathy: Mechanisms, Incidence and Clinical Management","year":2014,"authors":["Stroes, E.","Thompson, P.","Corsini, A."],
     "abstract":"Statin-associated muscle symptoms (SAMS) affect 5-10% of statin users in real-world practice versus 1-5% in clinical trials, with the discrepancy attributed to trial selection criteria and nocebo effects. CK elevation >10x ULN (rhabdomyolysis) occurs in 1 per 10,000 patient-years. SAMS are dose-dependent, more common with lipophilic statins, and risk increases with age, renal impairment, and drug interactions. Coenzyme Q10 supplementation has shown inconsistent benefit in RCTs. Understanding SAMS mechanisms is essential for maintaining therapeutic adherence."},
    {"title":"Statin Therapy and Risk of Incident Diabetes: Meta-Analysis","year":2010,"authors":["Sattar, N.","Preiss, D.","Murray, H."],
     "abstract":"Meta-analysis of 13 statin trials (n=91,140) found statin therapy associated with a 9% increase in diabetes risk (OR 1.09, 95% CI 1.02-1.17). The excess risk was most apparent with high-intensity statins. For every 255 patients treated with statins for 4 years, one additional diabetes case occurs. Cardiovascular risk reductions for primary prevention in high-risk individuals substantially outweigh the diabetogenic effect, but this trade-off is less favorable in low-risk populations."},
    {"title":"Statins in Women: Evidence for Cardiovascular Benefit","year":2017,"authors":["Cholesterol Treatment Trialists Collaboration"],
     "abstract":"Among 46,675 women in 27 statin trials, statin therapy reduced major vascular events by 16% per 1 mmol/L LDL reduction (rate ratio 0.84, 95% CI 0.78-0.91), a benefit not significantly different from men (rate ratio 0.78). For women without prior vascular disease (primary prevention), the proportional benefit was similar. Critics have noted that women are underrepresented in trials and that observational data suggest smaller real-world benefits. The evidence supports offering statins to women meeting standard cardiovascular risk thresholds."},
    {"title":"Statins and Cognitive Function: A Systematic Review","year":2015,"authors":["Ott, B.","Daiello, L.","Dahabreh, I."],
     "abstract":"Prospective cohort studies show mixed results: some find statin use associated with 29% reduced dementia risk while others find no effect. RCT evidence (PROSPER, ASTRONOMER) found no cognitive benefit or harm over 3-5 years. Case reports to FDA of statin-associated memory problems prompted a label change but remain anecdotal. The cognitive effects of statins likely depend on CNS penetration, timing of exposure, and underlying genetic risk. Current evidence does not support withholding statins due to dementia concern, but very long-term effects remain unknown."},
    {"title":"Statin Prescribing in the Elderly: Benefit and Harm Above Age 75","year":2020,"authors":["Orkaby, A.","Driver, J.","Ho, Y."],
     "abstract":"For adults over 75 without prior cardiovascular events, the benefit of statin initiation is uncertain. The 2016 ACC/AHA guidelines note insufficient evidence for this group. Observational studies show conflicting results, complicated by frailty, polypharmacy, and competing mortality risks. The STAREE trial (ongoing) will provide the first prospective evidence. Current practice involves individual assessment weighing likely cardiovascular benefit against pill burden, drug interactions, and patient life expectancy and values."},
    {"title":"Statin Discontinuation and Cardiovascular Events: A Nationwide Cohort","year":2017,"authors":["Yebyo, H.","Aschmann, H.","Kaufmann, M."],
     "abstract":"Among 53,320 primary prevention statin users followed for 9 years, discontinuation was associated with a 46% increase in major cardiovascular events (HR 1.46, 95% CI 1.39-1.53) after adjustment for confounders. Adherence below 80% attenuated benefits substantially. The largest adherence gap was observed in the first 6 months of therapy, suggesting that early side effect management is critical for long-term benefit."},
    {"title":"Rosuvastatin to Prevent Vascular Events in Men and Women with Elevated C-Reactive Protein","year":2008,"authors":["Ridker, P.","Danielson, E.","Fonseca, F."],
     "abstract":"The JUPITER trial enrolled 17,802 apparently healthy adults with LDL below 130 mg/dL and elevated high-sensitivity CRP (>=2.0 mg/L). Rosuvastatin 20mg reduced the incidence of major cardiovascular events by 44% and all-cause mortality by 20% compared to placebo after median 1.9 years. The trial was stopped early for benefit. Critics noted that the absolute risk reduction was modest (NNT approximately 95 for 5 years), the population was highly selected, and early stopping may have inflated effect sizes."},
    {"title":"Statins and Cancer Risk: A Meta-Analysis","year":2006,"authors":["Bonovas, S.","Filioussi, K.","Sitaras, N."],
     "abstract":"Meta-analysis of 26 randomized trials including 73,196 patients found no statistically significant association between statin therapy and overall cancer incidence (RR 1.02, 95% CI 0.97-1.07). Site-specific analyses showed no consistent pattern of increased or decreased cancer risk. Observational studies suggesting cancer protection are likely attributable to healthy user bias and surveillance bias. Statins should not be used for cancer prevention outside of clinical trials."},
    {"title":"Prevention of Coronary Heart Disease with Pravastatin in Men with Hypercholesterolemia (WOSCOPS)","year":1995,"authors":["Shepherd, J.","Cobbe, S.","Ford, I."],
     "abstract":"The West of Scotland Coronary Prevention Study enrolled 6,595 men with elevated LDL and no prior myocardial infarction. Pravastatin 40mg reduced coronary events by 31% (p<0.001) and total mortality by 22% (p=0.051) over 4.9 years. The NNT to prevent one major event was 31. This landmark trial established primary prevention as a viable statin indication. A 20-year follow-up found sustained mortality benefit, suggesting legacy effects of early statin therapy."},
    {"title":"Effectiveness of Statin Therapy in Adults with Coronary Heart Disease","year":2002,"authors":["Fonarow, G.","French, W.","Parsons, L."],
     "abstract":"In a national registry of 47,418 patients hospitalized for coronary heart disease, statin use prior to admission was associated with 47% lower in-hospital mortality (OR 0.53, 95% CI 0.46-0.60). Statin initiation before hospital discharge was associated with improved one-year survival. These findings support both primary and secondary prevention statin use and suggest that statin underuse represents a significant patient safety concern in the acute cardiovascular care setting."},
    {"title":"Statin Safety and Tolerability: An Expert Consensus","year":2007,"authors":["Armitage, J."],
     "abstract":"Statins are among the most extensively studied medications in history. Serious adverse effects are rare: rhabdomyolysis (1 per 10,000 patient-years), hepatotoxicity (3-fold LFT elevation in 1-3% requiring monitoring), and new-onset diabetes (1 per 1,000 patient-years for high-intensity). The majority of statin-associated myalgia complaints (approximately 70%) are not causally related to statin therapy based on blinded rechallenge studies. Patient communication about absolute (not relative) risks and benefits is essential for informed consent and long-term adherence."},
    {"title":"Meta-Analysis of Statin Effects in Women versus Men","year":2012,"authors":["Petretta, M.","Costanzo, P.","Perrone-Filardi, P."],
     "abstract":"Pooling data from 18 primary prevention trials (n=56,934), we found that statins reduced major cardiovascular events comparably in women (RR 0.84) and men (RR 0.78), with the difference not statistically significant (p=0.38 for interaction). However, all-cause mortality was not significantly reduced in women in primary prevention (RR 0.95, CI 0.86-1.05), while it was in men (RR 0.92, CI 0.87-0.97). These findings suggest that primary prevention statin decisions in women should be based on individual cardiovascular risk rather than sex alone."},
    {"title":"Cardiovascular Event Reduction with Evolocumab in Patients with Established Cardiovascular Disease","year":2017,"authors":["Sabatine, M.","Giugliano, R.","Keech, A."],
     "abstract":"The FOURIER trial demonstrated that evolocumab (PCSK9 inhibitor) reducing LDL from 92 to 30 mg/dL further reduced major cardiovascular events by 15% (p<0.001) in patients already on statin therapy. This provides strong evidence for the LDL hypothesis: lower is better, with no threshold effect apparent. The results support aggressive LDL lowering and validate LDL-C as a modifiable causal risk factor. However, all-cause mortality was not significantly reduced, and cost-effectiveness remains an issue at current PCSK9 inhibitor pricing."},
  ],

  "minwage": [
    {"title":"Minimum Wages and Employment: A Case Study of the Fast-Food Industry","year":1994,"authors":["Card, D.","Krueger, A."],
     "abstract":"We examine the effect of New Jersey's 1992 minimum wage increase on employment in the fast-food industry, using Pennsylvania as a control. Employment in NJ grew relative to PA following the minimum wage increase. Traditional models predict employment losses; our finding is inconsistent with the competitive labor market model and suggests employers have monopsony power, or that higher wages reduce turnover sufficiently to offset wage costs. This paper sparked a major empirical debate about minimum wage employment effects."},
    {"title":"Minimum Wages and Employment: A New Look at the Evidence","year":2010,"authors":["Dube, A.","Lester, T.","Reich, M."],
     "abstract":"Using county pairs straddling state borders as a quasi-experimental control, we find no detectable employment effects of minimum wage increases in restaurant and retail industries. Our approach controls for local economic trends that prior studies fail to isolate. Employment in border-county pairs on the lower-wage side of the state line was not systematically higher than on the higher-wage side after minimum wage increases. This suggests that small minimum wage increases do not cause employment losses in low-wage industries."},
    {"title":"Minimum Wages and Employment: Evidence from a Regression Discontinuity Design","year":2014,"authors":["Neumark, D.","Salas, J.","Wascher, W."],
     "abstract":"Using a regression discontinuity design based on state-level minimum wage legislation, we find negative employment effects for teenagers (elasticity -0.1 to -0.2) and restaurant workers (elasticity -0.05 to -0.15). Effects are larger for small businesses and in low-wage areas where the minimum wage represents a larger bite. Our results differ from Dube et al. (2010) because county-pair designs fail to account for local minimum wage variation and confound policy effects with local economic conditions."},
    {"title":"Minimum Wage, Labor Market Monopsony and the Gender Pay Gap","year":2018,"authors":["Autor, D.","Manning, A.","Smith, C."],
     "abstract":"Monopsony power among minimum wage employers explains why employment losses are smaller than competitive models predict. When employers face upward-sloping labor supply, a minimum wage increase above the competitive equilibrium can increase both wages and employment simultaneously. We estimate that monopsony explains roughly one-third of the wage gap between men and women in low-wage industries, and that minimum wages disproportionately benefit women. Model-consistent evidence suggests minimum wages should be set at roughly 50-60% of the local median wage."},
    {"title":"Seattle's Minimum Wage Experience 2015-16","year":2017,"authors":["Jardim, E.","Long, M.","Plotnick, R."],
     "abstract":"Using administrative payroll data from Washington State, we find that Seattle's minimum wage increase to $13/hour reduced hours worked by low-wage workers by 9% and reduced monthly income by $125. These effects are larger than prior studies because we use hours data rather than employment counts; workers keep their jobs but have hours cut. Our methods are disputed by some researchers, but the magnitude of hours effects suggests that employment counts alone understate the labor demand response to minimum wages."},
    {"title":"Minimum Wage and Poverty: The Effects on the Poor","year":2016,"authors":["Dube, A."],
     "abstract":"Using the March CPS from 1990-2012, we find that a 10% minimum wage increase reduces the poverty rate by 2.4% (elasticity -0.24). Effects are concentrated among individuals just above and below the poverty line. Families in the bottom income quintile gain substantially, with gains concentrated among single-parent households. We reconcile apparent inconsistency between modest employment effects and substantial poverty effects by noting that workers in poor families are more likely to earn the minimum wage."},
    {"title":"Minimum Wages and Firm Entry and Exit","year":2020,"authors":["Draca, M.","Machin, S.","Van Reenen, J."],
     "abstract":"UK minimum wage increases led to higher wages without large employment effects, but reduced profits in low-wage industries. Firm exit rates increased modestly in high-exposure industries while entry rates declined. Net firm destruction accounts for 15-20% of the measured labor demand response to minimum wages, with employment per remaining firm holding constant. Large businesses are better able to absorb minimum wage increases than small businesses."},
    {"title":"The Effect of Minimum Wages on Low-Wage Jobs","year":2019,"authors":["Cengiz, D.","Dube, A.","Lindner, A."],
     "abstract":"Using the full distribution of wages across all US states from 1979-2016, we estimate that minimum wage increases raised wages of low-wage workers substantially without reducing the number of low-wage jobs. We find a small average effect on employment of -0.002 (not statistically significant). The job losses at wages below the new minimum were nearly entirely offset by job gains just above the new minimum, suggesting wage compression rather than employment reduction as the primary margin of adjustment."},
    {"title":"Effects of a $15 Minimum Wage in New York State","year":2021,"authors":["Reich, M.","Jacobs, K.","Bernhardt, A."],
     "abstract":"Analysis of restaurant employment in New York's phased-in $15 minimum wage found no significant employment loss in restaurants through the phase-in period. Employment and hours in New York restaurants tracked closely with neighboring states that did not increase their minimum wages. The phase-in allowed businesses to adjust through productivity improvements, price increases of approximately 0.7%, and modest profit compression. The evidence supports a $15 minimum wage phased in gradually as unlikely to produce large employment losses."},
    {"title":"Minimum Wage Effects Across State Borders: Estimates Using Contiguous Counties","year":2010,"authors":["Dube, A.","Lester, W.","Reich, M."],
     "abstract":"We examine the effect of minimum wages on restaurant employment using a dataset of 288 contiguous county pairs straddling state borders from 1990-2006. Within these county pairs, minimum wage increases have no detectable employment effects even at the industry level. The key identifying assumption is that contiguous counties share similar economic trends; our tests support this assumption. Prior studies using all-state comparisons are biased by differential trends between high- and low-minimum-wage states."},
    {"title":"Who Pays for the Minimum Wage?","year":2014,"authors":["Harasztosi, P.","Lindner, A."],
     "abstract":"Examining Hungary's large minimum wage increase in 2001, we find that approximately 75% of the cost was passed on to consumers through price increases, 12% through profit compression, and 13% through productivity gains. Employment effects were negligible despite a 60% minimum wage increase. The evidence suggests that minimum wages at moderate levels can be absorbed largely through consumer prices without significant employment effects, particularly in non-traded service sectors where competition is limited."},
    {"title":"The Congressional Budget Office Analysis of Minimum Wage Increases","year":2021,"authors":["CBO"],
     "abstract":"CBO estimates that raising the federal minimum wage to $15 by 2025 would lift 900,000 people out of poverty and increase wages for 17 million workers directly and 10 million workers through spillover effects. It would also reduce employment by 1.4 million (90% confidence interval: near zero to 2.7 million job loss). The uncertainty range reflects genuine scientific disagreement about minimum wage employment effects. Higher-income workers would see indirect wage increases due to compression; the net budgetary effect is slightly positive through reduced poverty program spending."},
    {"title":"Do Minimum Wages Reduce Employment? Evidence from OECD Countries","year":2019,"authors":["Neumark, D.","Shirley, P."],
     "abstract":"Cross-country analysis of minimum wages in 31 OECD countries from 1975-2015 finds negative but heterogeneous employment effects. Countries with minimum wages above 50% of the median wage show larger disemployment effects. Youth (15-24) employment is more sensitive than adult employment. We find no evidence that institutional factors (such as union density or employment protection) systematically moderate minimum wage effects across countries. The cross-country evidence supports the existence of employment effects, contradicting the no-effect consensus."},
    {"title":"Minimum Wages and Income Inequality: Evidence from United States","year":2016,"authors":["Autor, D.","Manning, A.","Smith, C."],
     "abstract":"Minimum wages reduced wage inequality in the United States substantially from 1979-2012. We find that declining real minimum wages explain about 30-40% of the increase in wage inequality at the bottom of the distribution during this period. Recent state minimum wage increases have reversed some of this trend. The effect on inequality is larger than employment effects, suggesting minimum wages are a powerful tool for wage compression even when employment effects are modest."},
    {"title":"Myth or Measurement: The New Economics of the Minimum Wage","year":1995,"authors":["Card, D.","Krueger, A."],
     "abstract":"We revisit the empirical evidence on minimum wages and find it inconsistent with the standard competitive model. Using both time-series and cross-sectional data, minimum wage increases do not reduce employment in low-wage industries. We attribute this to labor market imperfections including search frictions, monopsony, and efficiency wages. Our findings were vigorously contested by Neumark and Wascher using different data and methods, establishing a major empirical debate that continues in the literature today with both camps claiming methodological superiority."},
  ],

  "climate": [
    {"title":"Carbon Dioxide and Climate: A Scientific Assessment (Charney Report)","year":1979,"authors":["Charney, J.","Arakawa, A.","Baker, D."],
     "abstract":"A National Academy of Sciences assessment concluded that doubling atmospheric CO2 would produce a global mean warming of approximately 3 degrees Celsius, with a plausible range of 1.5-4.5 degrees. This assessment synthesized results from general circulation models and radiative transfer calculations. The study identified water vapor and ice-albedo feedbacks as primary amplifiers of the initial radiative forcing. The 1.5-4.5 degree likely range persisted in subsequent IPCC assessments for four decades, reflecting the difficulty of constraining climate sensitivity."},
    {"title":"IPCC AR6 Working Group I: The Physical Science Basis Summary","year":2021,"authors":["Masson-Delmotte, V.","Zhai, P.","Pirani, A."],
     "abstract":"The Sixth Assessment Report of the IPCC concluded that human influence has warmed the atmosphere, ocean, and land unequivocally. Global surface temperature has increased faster since 1970 than in any other 50-year period over at least the last 2000 years. Climate sensitivity (ECS) is assessed to be 2.5-4.0 degrees Celsius with a best estimate of 3.0 degrees, narrowing from the previous 1.5-4.5 range. Many changes are unprecedented in hundreds to thousands of years, and some (sea level rise, ocean acidification) are irreversible on centuries-long timescales."},
    {"title":"The Impact of Recent Forcing and Ocean Heat Uptake Data on Estimates of Climate Sensitivity","year":2018,"authors":["Lewis, N.","Curry, J."],
     "abstract":"Using updated estimates of forcing and ocean heat uptake from 2000-2016, we derive an energy-balance constrained estimate of equilibrium climate sensitivity of 1.66 degrees Celsius (5-95% range: 1.15-2.70). This estimate is substantially lower than IPCC's likely range. We argue that the discrepancy reflects over-reliance on climate models in IPCC assessments relative to observational constraints. Our lower estimate implies reduced urgency for rapid decarbonization but does not eliminate the need for emissions reductions over the coming century."},
    {"title":"An Assessment of Earth's Climate Sensitivity Using Multiple Lines of Evidence","year":2020,"authors":["Sherwood, S.","Webb, M.","Annan, J."],
     "abstract":"Using three independent lines of evidence—process understanding, historical warming, and paleoclimate—we derive a constrained estimate of equilibrium climate sensitivity of 2.3-4.7 degrees (5-95% range), with a median of 3.1 degrees. The assessment rules out ECS below 2 degrees Celsius with high confidence and ECS above 4.5 degrees with moderate confidence. Reconciliation of historical forcing estimates with paleoclimate evidence eliminates the low-end tail previously allowed by energy balance methods alone. This assessment informed the IPCC AR6 likely range."},
    {"title":"Anthropogenic and Natural Radiative Forcing (IPCC AR5 Chapter 8)","year":2013,"authors":["Myhre, G.","Shindell, D.","Breon, F."],
     "abstract":"The total anthropogenic radiative forcing over the industrial era is estimated at 2.29 W/m2 (likely range 1.13-3.33 W/m2), dominated by CO2 (1.68 W/m2). Aerosol forcing remains the largest uncertainty, estimated at -0.9 W/m2 (range -1.9 to -0.1 W/m2). Natural forcing (solar and volcanic) contributed approximately 0.1 W/m2 since 1750. The large aerosol forcing uncertainty propagates directly into climate sensitivity uncertainty, since observational estimates of climate sensitivity require accurate forcing inputs."},
    {"title":"Earth's Energy Imbalance: Confirmation and Implications","year":2005,"authors":["Hansen, J.","Nazarenko, L.","Ruedy, R."],
     "abstract":"Ocean heat content measurements confirm an energy imbalance of approximately 0.85 W/m2 at the top of atmosphere, consistent with climate model projections for current greenhouse gas forcing. This imbalance implies a committed future warming of approximately 0.6 degrees Celsius even if greenhouse gas concentrations were held constant. The large heat capacity of the ocean means current warming underestimates equilibrium warming. Reducing aerosol forcing (clean air policy) would accelerate the realization of committed warming by removing a masking effect."},
    {"title":"Otto et al: Energy Budget Constraints on Climate Response","year":2013,"authors":["Otto, A.","Otto, F.","Boucher, O."],
     "abstract":"Using observed warming, ocean heat uptake, and forcing estimates through 2009, we derive energy-budget constrained estimates of transient climate response (TCR) of 1.3 degrees Celsius (5-95%: 0.9-2.0) and equilibrium climate sensitivity (ECS) of 2.0 degrees (5-95%: 1.2-3.9). These estimates are at the lower end of IPCC assessments. The discrepancy with higher model-based estimates may reflect model deficiencies in representing aerosol forcing, ocean heat uptake, or natural variability. These results do not contradict human-caused warming but suggest slower near-term warming than model medians imply."},
    {"title":"Emergent Constraint on Equilibrium Climate Sensitivity from Global Temperature Variability","year":2018,"authors":["Cox, P.","Huntingford, C.","Williamson, M."],
     "abstract":"Exploiting the relationship between global temperature variability and equilibrium climate sensitivity across CMIP5 models, we derive an observationally constrained ECS estimate of 2.8 degrees (66% range: 2.2-3.4). This emergent constraint method uses observed interannual variability to evaluate which models best match reality. Our estimate sits within the IPCC likely range but near the lower end, and provides an independent estimate that does not depend on accurate aerosol forcing estimation. Multiple emergent constraints have since been proposed with varying results."},
    {"title":"The Equilibrium Climate Sensitivity","year":2008,"authors":["Knutti, R.","Hegerl, G."],
     "abstract":"We review the constraints on equilibrium climate sensitivity from multiple independent lines of evidence: paleoclimate records, instrumental observations, and models. All consistent evidence suggests ECS is most likely 2-4.5 degrees Celsius, with the probability of ECS exceeding 4.5 degrees being 5-17% across different estimation methods. The lower bound of approximately 1.5 degrees is well established by physics. The upper tail remains poorly constrained due to the difficulty of observationally constraining cloud feedbacks. We recommend treating climate sensitivity as a distribution, not a single value, in policy analysis."},
    {"title":"Why Is Climate Sensitivity So Unpredictable?","year":2007,"authors":["Roe, G.","Baker, M."],
     "abstract":"We demonstrate that the uncertainty in climate sensitivity arises from a near-cancellation of large feedbacks. When feedbacks sum close to 1, small changes in feedback estimates produce large changes in sensitivity through the amplification relationship. This mathematical structure means that constraining ECS below 4.5 degrees requires extremely accurate feedback measurements, while reducing the upper tail requires reducing feedback uncertainty by an order of magnitude. Our analysis implies that large reductions in climate sensitivity uncertainty may not be achievable with current observational methods."},
    {"title":"Implications for Climate Sensitivity from the Response to Individual Forcings","year":2016,"authors":["Marvel, K.","Schmidt, G.","Miller, R."],
     "abstract":"Different forcing agents produce different patterns of surface temperature change, and pattern effects influence apparent climate sensitivity estimates. Efficacy factors estimated from GISS ModelE2 show that aerosol forcing is 45% more effective at producing surface temperature change than CO2 forcing. Correcting for forcing efficacy increases effective climate sensitivity estimates derived from historical observations. If correct, energy-budget estimates of ECS that treat all forcings as equivalent may underestimate true ECS by 0.3-0.8 degrees Celsius."},
    {"title":"Attribution of Extreme Weather Events to Human Influence on Climate","year":2016,"authors":["National Academies of Sciences, Engineering, and Medicine"],
     "abstract":"Attribution science has advanced to the point where it is possible to quantify how climate change has affected the probability or magnitude of specific types of extreme events. Heat extremes are the most reliably attributed category, with confidence that anthropogenic forcing has made extreme heat events more frequent globally. Attribution of precipitation extremes is more uncertain due to natural variability and observational limitations. The field has rapidly advanced but still faces challenges of observational length, model dependence, and communicating probabilistic statements to the public."},
    {"title":"Atmospheric Aerosols: Multidisciplinary Review","year":2013,"authors":["Boucher, O.","Randall, D.","Artaxo, P."],
     "abstract":"Aerosol-radiation interactions (direct forcing) and aerosol-cloud interactions (indirect forcing) together constitute the largest source of uncertainty in radiative forcing estimates. The best estimate of total aerosol forcing is -0.9 W/m2, but the likely range spans -1.9 to -0.1 W/m2. This uncertainty limits the ability to derive climate sensitivity from the historical temperature record. Recent satellite measurements of cloud properties have constrained the indirect aerosol effect somewhat, but large uncertainty remains. Better aerosol observations are the single most important requirement for reducing climate sensitivity uncertainty."},
    {"title":"Climate Sensitivity Estimates from Modern Global Observations","year":2019,"authors":["Zelinka, M.","Myers, T.","McCoy, D."],
     "abstract":"CMIP6 models show a wider range of equilibrium climate sensitivity (1.8-5.6 degrees) than CMIP5 (2.1-4.7 degrees), largely due to changes in extratropical low-cloud feedbacks. The higher end of CMIP6 ECS estimates are inconsistent with observational constraints from historical warming. A subset of models with ECS above 5 degrees can be identified as outliers using modern observational metrics. This finding is significant: some of the highest-end climate projections from CMIP6 are observationally disfavored, narrowing the effective likely range."},
    {"title":"Detection and Attribution of Climate Change: From Global to Regional","year":2013,"authors":["Bindoff, N.","Stott, P.","AchutaRao, K."],
     "abstract":"The detection and attribution of climate change has advanced substantially since the Third Assessment Report. It is now extremely likely (95% confidence) that human influence has been the dominant cause of observed warming since 1950. Attribution extends beyond global mean temperature to ocean warming, sea level rise, Arctic sea ice decline, and changes in extreme events. Regional attribution is less certain due to larger contributions from natural variability at smaller spatial scales. The evidence base now includes multiple independent observation types and attribution methods."},
  ],

  "diet_cvd": [
    {"title":"The Diet and 15-Year Death Rate in the Seven Countries Study","year":1986,"authors":["Keys, A.","Menotti, A.","Karvonen, M."],
     "abstract":"The Seven Countries Study, following 12,763 middle-aged men across seven countries for 15 years, found that saturated fat intake was strongly correlated with coronary heart disease mortality rates (r=0.84). Countries with high saturated fat intakes (Finland, USA) had the highest CHD death rates; those with low saturated fat intakes (Japan, Greece) had the lowest. This ecological study provided the foundational evidence for the diet-heart hypothesis and shaped dietary guidelines for decades. Critics subsequently charged that Keys selectively included countries supporting his hypothesis."},
    {"title":"Meta-Analysis of Prospective Cohort Studies Evaluating the Association of Saturated Fat with Cardiovascular Disease","year":2010,"authors":["Siri-Tarino, P.","Sun, Q.","Hu, F."],
     "abstract":"A meta-analysis of 21 prospective cohort studies (n=347,747, 5-23 year follow-up) found no significant association between saturated fat intake and cardiovascular disease (relative risk 1.07, 95% CI 0.96-1.19) or coronary heart disease (RR 1.03, 0.90-1.17). The authors conclude that available prospective evidence does not support dietary saturated fat being associated with increased CVD risk. This paper was widely cited to challenge dietary guidelines but was criticized for not considering the replacement nutrient when saturated fat is reduced."},
    {"title":"Association of Dietary, Circulating, and Supplement Fatty Acids with Coronary Risk","year":2014,"authors":["Chowdhury, R.","Warnakula, S.","Kunutsor, S."],
     "abstract":"A meta-analysis encompassing 72 unique studies found no significant association between dietary saturated fat and coronary disease (RR 1.02, 95% CI 0.97-1.07). Similarly, circulating saturated fatty acid biomarkers showed no significant association with coronary outcomes. The study suggested that omega-3 fatty acids (DHA, EPA, DPA) were associated with lower coronary risk. The findings were controversial; critics argued the analysis failed to account for the macronutrient replacing saturated fat, potentially masking true effects."},
    {"title":"Reduction in Saturated Fat Intake for Cardiovascular Disease (Cochrane Review)","year":2015,"authors":["Hooper, L.","Martin, N.","Abdelhamid, A."],
     "abstract":"This Cochrane review of 15 randomized controlled trials (n=59,000) found that reducing saturated fat intake reduced cardiovascular events by 17% (RR 0.83, 95% CI 0.72-0.96) but did not significantly reduce total mortality or cardiovascular mortality. The effect on cardiovascular events was driven primarily by studies replacing saturated fat with polyunsaturated fat. Trials replacing saturated fat with carbohydrate showed little benefit, supporting the importance of the replacement nutrient. The review concluded that reducing saturated fat and replacing with polyunsaturated fat is beneficial."},
    {"title":"Major Types of Dietary Fat and Risk of Coronary Heart Disease","year":2009,"authors":["Jakobsen, M.","O'Reilly, E.","Heitmann, B."],
     "abstract":"In a pooled analysis of 11 American and European cohort studies (n=344,696), we estimated the association between different dietary fat types and coronary heart disease. Replacing 5% of energy from saturated fats with equivalent energy from polyunsaturated fats was associated with a 13% lower CHD risk (RR 0.87, 95% CI 0.77-0.97). Replacing saturated fats with monounsaturated fats was not significantly associated with CHD risk. Replacing saturated fats with refined carbohydrates was not associated with lower CHD risk. These findings support the importance of the replacement macronutrient."},
    {"title":"Trans-Fatty Acids and Cardiovascular Disease","year":2006,"authors":["Mozaffarian, D.","Katan, M.","Ascherio, A."],
     "abstract":"Industrial trans-fatty acids (elaidic acid from partial hydrogenation) raise LDL cholesterol, lower HDL cholesterol, and promote inflammation and endothelial dysfunction. Meta-analyses show that for each 2% increase in energy from trans fats, coronary heart disease risk increases by 23-28%. The evidence for harm from industrial trans fats is stronger than for saturated fats. Ruminant trans fats (from dairy and beef, primarily vaccenic and conjugated linoleic acid) do not appear to carry the same risk and may be neutral or beneficial. Trans fat elimination from the food supply is strongly justified by evidence."},
    {"title":"Associations of Fats and Carbohydrate Intake with Cardiovascular Disease and Mortality in 18 Countries: PURE Study","year":2017,"authors":["Dehghan, M.","Mente, A.","Zhang, X."],
     "abstract":"The PURE prospective cohort study of 135,335 adults across 18 countries found that high carbohydrate intake was associated with higher mortality, while total fat and each fat type were associated with lower mortality. High saturated fat was associated with lower stroke risk. These findings challenge existing dietary guidelines that emphasize low fat intake. Critics argued that confounding by food insecurity and underconsumption in low-income countries drove results, and that the observational design cannot establish causation."},
    {"title":"Primary Prevention of Cardiovascular Disease with a Mediterranean Diet: PREDIMED","year":2018,"authors":["Estruch, R.","Ros, E.","Salas-Salvado, J."],
     "abstract":"The PREDIMED trial randomized 7,447 high-cardiovascular-risk adults to Mediterranean diet with olive oil, Mediterranean diet with nuts, or low-fat control diet. After 4.8 years, Mediterranean diet groups had 31% lower rates of major cardiovascular events. The trial was retracted and republished due to randomization irregularities; the reanalysis showed an HR of 0.76 (0.63-0.90) for Mediterranean diet with olive oil, slightly attenuated but still significant. The trial provides the strongest RCT evidence for a dietary pattern on cardiovascular outcomes."},
    {"title":"Dietary Fat Quality and Risk of Sudden Cardiac Death: Women's Health Initiative","year":2006,"authors":["Howard, B.","Van Horn, L.","Hsia, J."],
     "abstract":"The Women's Health Initiative Dietary Modification Trial randomized 48,835 postmenopausal women to a low-fat dietary intervention (target 20% of calories from fat) or usual diet. After 8.1 years, the intervention group achieved a modest fat reduction (from 38% to 29% of calories) but showed no significant reduction in coronary heart disease (HR 0.97, 95% CI 0.90-1.06) or stroke (HR 1.02, 0.90-1.17). The trial was criticized for poor adherence, insufficient fat reduction, and not specifying the replacement for fat calories (often refined carbohydrates)."},
    {"title":"Saturated Fatty Acids and Plasma Lipids","year":2014,"authors":["Schwab, U.","Lauritzen, L.","Tholstrup, T."],
     "abstract":"Different saturated fatty acids have distinct effects on plasma lipid profiles. Lauric acid (C12:0, coconut oil) and myristic acid (C14:0, dairy) raise both LDL and HDL cholesterol substantially. Palmitic acid (C16:0, palm oil, meat) raises LDL with smaller HDL increase. Stearic acid (C18:0, chocolate) has a neutral effect on LDL and may slightly lower it. The cardiovascular implications differ: acids that raise both LDL and HDL may have lower net cardiovascular risk than those raising LDL without raising HDL. Food matrix effects further modulate these responses."},
    {"title":"Saturated Fats and CVD: Replacing What with What?","year":2017,"authors":["Briggs, M.","Petersen, K.","Kris-Etherton, P."],
     "abstract":"The key question in saturated fat and cardiovascular disease is not whether to reduce saturated fat, but what to replace it with. Replacing saturated fat with polyunsaturated fat (omega-6 linoleic acid) reduces LDL and cardiovascular events. Replacing with monounsaturated fat has inconsistent effects. Replacing with refined carbohydrates and added sugars increases triglycerides and may worsen cardiovascular risk. Replacing with whole grains and dietary fiber reduces cardiovascular risk. Dietary guidelines should specify replacements, not merely recommend reduction."},
    {"title":"A Systematic Review of Dietary Adherence and Cardiovascular Disease Prevention","year":2009,"authors":["Mente, A.","de Koning, L.","Shannon, H."],
     "abstract":"Systematic review of 146 prospective cohort and 43 randomized trial studies found strong evidence for cardiovascular benefit from Mediterranean and prudent dietary patterns, and harmful effects from Western dietary patterns high in red and processed meat. Evidence was moderate for the diet-heart hypothesis linking dietary fat to cardiovascular outcomes in RCTs. The authors conclude that dietary pattern evidence is stronger than individual nutrient evidence, and that guidelines should emphasize patterns over single nutrients."},
    {"title":"Olive Oil Intake and Risk of Cardiovascular Disease and Mortality: A Systematic Review","year":2015,"authors":["Guasch-Ferre, M.","Hu, F.","Martinez-Gonzalez, M."],
     "abstract":"A meta-analysis of 32 studies found that higher olive oil intake was associated with 9% lower cardiovascular mortality (RR 0.91, 95% CI 0.85-0.96) and 11% lower stroke risk. Extra virgin olive oil showed larger protective effects than refined olive oil, suggesting that polyphenols beyond oleic acid contribute to cardiovascular protection. These associations were independent of Mediterranean dietary pattern adherence, suggesting olive oil has specific cardiovascular effects beyond dietary pattern synergy."},
    {"title":"Saturated Fats and Health: A Reassessment and Proposal for Food-Based Recommendations","year":2020,"authors":["Astrup, A.","Magkos, F.","Bier, D."],
     "abstract":"We argue that the totality of evidence does not support a direct link between saturated fat intake per se and cardiovascular disease when dairy foods are considered separately from meat. Fermented dairy products (cheese, yogurt) show neutral or protective cardiovascular associations despite high saturated fat content. The food matrix in which saturated fats are consumed determines metabolic effects. Existing dietary guidelines to reduce saturated fat intake may need revision to account for food matrix effects. Critics noted this paper received funding from dairy industry sources."},
    {"title":"Marine n-3 Fatty Acids and Prevention of Cardiovascular Disease: Meta-Analysis","year":2018,"authors":["Aung, T.","Halsey, J.","Kromhout, D."],
     "abstract":"The ASCEND and ORIGIN trials of omega-3 supplementation found no significant reduction in cardiovascular events (ORIGIN: HR 0.98; ASCEND: RR 0.97), contradicting prior evidence. A meta-analysis of 10 trials found cardiovascular death reduction of 8% (RR 0.92, 95% CI 0.86-0.98) but no significant effect on myocardial infarction. The REDUCE-IT trial of high-dose EPA (4g/day icosapentaenoic acid) found 25% cardiovascular event reduction, but critics questioned whether the mineral oil placebo was truly inert, potentially exaggerating the apparent benefit."},
  ],
}

# ─── TF-IDF embeddings (no sentence-transformers dependency) ──────────────────
import math as _math

_idf_cache: dict = {}
_corpus_tokens: list = []
_vocab: list = []

def _tokenize(text: str) -> list:
    STOP = {"the","a","an","of","in","to","and","is","are","was","were","for",
            "with","that","this","it","be","by","on","at","from","as","or","but",
            "not","have","has","had","can","will","its","their","they","we","our",
            "more","than","some","when","does","also","both","into","such","each"}
    return [w for w in re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()
            if len(w) > 2 and w not in STOP]

def _build_idf(corpus: list) -> dict:
    N = len(corpus)
    df: dict = {}
    for doc in corpus:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    return {w: _math.log((N + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}

def _tfidf_vec(tokens: list, idf: dict) -> dict:
    tf: dict = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    n = len(tokens) or 1
    return {w: (cnt / n) * idf.get(w, 1.0) for w, cnt in tf.items()}

def embed(texts: list) -> list:
    global _idf_cache, _corpus_tokens, _vocab
    import numpy as np
    all_tokens = [_tokenize(t) for t in texts]
    if not _idf_cache or len(texts) > len(_corpus_tokens):
        _corpus_tokens = all_tokens
        _idf_cache = _build_idf(_corpus_tokens)
        _vocab = sorted(_idf_cache.keys())
    vecs = []
    for tokens in all_tokens:
        sparse = _tfidf_vec(tokens, _idf_cache)
        dense = np.array([sparse.get(w, 0.0) for w in _vocab], dtype=np.float32)
        norm = np.linalg.norm(dense)
        vecs.append((dense / (norm + 1e-9)).tolist())
    return vecs

def cosine(a: list, b: list) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── logging ──────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ─── Ollama API ───────────────────────────────────────────────────────────────
def ollama(system: str, user: str, max_tok: int = 800, as_json: bool = False):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": False,                                    # top-level for qwen3 thinking models
        "options": {"num_predict": max_tok, "temperature": 0.1},
        **({"format": "json"} if as_json else {}),
    }).encode()
    t0 = time.time()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            content = data.get("message", {}).get("content", "") or data.get("response", "")
            return content, round(time.time() - t0, 2)
        except Exception as e:
            log(f"[WARN] Ollama attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return "", round(time.time() - t0, 2)

def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ─── Semantic Scholar fetch ───────────────────────────────────────────────────
def fetch_abstracts(query: str, limit: int = 5) -> list:
    params = urllib.parse.urlencode({
        "query": query, "fields": "title,abstract,year,authors", "limit": limit})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EVIRAG-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            papers = []
            for p in data.get("data", []):
                if p.get("abstract") and len(p["abstract"]) > 100:
                    papers.append({
                        "title": p.get("title", ""),
                        "abstract": p["abstract"],
                        "year": p.get("year", 2000),
                        "authors": [a["name"] for a in p.get("authors", [])[:3]],
                    })
            return papers
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                log(f"[WARN] 429 rate-limit attempt {attempt+1}; backing off {wait:.1f}s")
                time.sleep(wait)
            else:
                return []
        except Exception as e:
            log(f"[WARN] fetch failed '{query}': {e}")
            return []
    return []


# ─── claim extraction ─────────────────────────────────────────────────────────
def extract_claims(text: str, source: str) -> list:
    user = (f'Extract all atomic factual claims from this scientific text. '
            f'Each claim must be: (1) one fact only, (2) self-contained.\n\n'
            f'Text: "{text[:1500]}"\n\n'
            f'Respond ONLY with valid JSON:\n'
            f'{{"claims":[{{"id":1,"text":"claim text","confidence":0.8}},...]}}'  )
    raw, _ = ollama("You are a precise scientific claim extractor. Output valid JSON only.", user,
                    max_tok=600, as_json=True)
    parsed = parse_json(raw)
    claims = parsed.get("claims", [])
    return [
        {"text": c.get("text","") if isinstance(c,dict) else str(c),
         "source": source,
         "confidence": c.get("confidence",0.8) if isinstance(c,dict) else 0.8}
        for c in claims if (c.get("text","") if isinstance(c,dict) else str(c)).strip()
    ]

def label_relation(ca: str, cb: str) -> dict:
    user = (f'Classify the relationship between these scientific claims.\n\n'
            f'Claim A: "{ca}"\nClaim B: "{cb}"\n\n'
            f'Output ONLY valid JSON:\n'
            f'{{"label":"SUPPORTS|CONTRADICTS|NEUTRAL","confidence":0.0-1.0}}')
    raw, t = ollama("You are a scientific NLI classifier. Output valid JSON only.", user,
                    max_tok=80, as_json=True)
    p = parse_json(raw)
    return {"label": p.get("label","NEUTRAL"), "confidence": p.get("confidence",0.5), "time": t}

def label_cda7(ca: str, cb: str) -> str:
    classes = "METHODOLOGICAL|POPULATION|TEMPORAL|OPERATIONAL|STATISTICAL|THEORETICAL|REPLICATION"
    user = (f'Two scientific claims contradict each other. Identify the PRIMARY cause.\n\n'
            f'Claim A: "{ca}"\nClaim B: "{cb}"\n\n'
            f'CDA-7 classes: {classes}\n'
            f'Output ONLY valid JSON: {{"label":"<one class>"}}')
    raw, _ = ollama("You are a scientific disagreement analyst. Output valid JSON only.", user,
                    max_tok=60, as_json=True)
    p = parse_json(raw)
    lbl = p.get("label","METHODOLOGICAL")
    valid = set(classes.split("|"))
    return lbl if lbl in valid else "METHODOLOGICAL"


# ─── metrics ─────────────────────────────────────────────────────────────────
def compute_vc_keywords(response: str, gt_views: list) -> float:
    resp_l = response.lower()
    covered = 0
    for view in gt_views:
        words = [w.lower() for w in re.findall(r'\b[a-z]{4,}\b', view) if w.lower() not in {
            'that','this','with','from','have','been','they','their','which','about',
            'more','than','some','when','does','also','both','into','such','each'}]
        key_words = list(dict.fromkeys(words))[:10]
        hits = sum(1 for kw in key_words if kw in resp_l)
        covered += int(hits >= 3)
    return covered / len(gt_views) if gt_views else 0.0

def compute_vc_embedding(response: str, gt_views: list) -> float:
    DELTA = 0.3
    resp_emb = embed([response])[0]
    view_embs = embed(gt_views)
    covered = sum(1 for ve in view_embs if (1 - cosine(resp_emb, ve)) <= DELTA)
    return covered / len(gt_views) if gt_views else 0.0

def compute_cr(response: str, gt_contradictions: list) -> float:
    if not gt_contradictions:
        return 0.0
    resp_l = response.lower()
    detected = 0
    for (term_a, term_b) in gt_contradictions:
        words_a = [w for w in term_a.split() if len(w) > 4][:3]
        words_b = [w for w in term_b.split() if len(w) > 4][:3]
        has_a = any(w in resp_l for w in words_a)
        has_b = any(w in resp_l for w in words_b)
        detected += int(has_a and has_b)
    return detected / len(gt_contradictions)

def compute_ccs(response: str, gt_views: list) -> float:
    if not gt_views:
        return 0.0
    import numpy as np
    resp_emb = np.array(embed([response])[0])
    view_embs = np.array(embed(gt_views))
    mean_emb = view_embs.mean(axis=0)
    resp_norm = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
    mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
    return float(1 - np.dot(resp_norm, mean_norm))

def compute_cce(response: str, domain_gt_conf: float) -> float:
    resp_l = response.lower()
    if "polarized" in resp_l or ("low" in resp_l and "confidence" in resp_l):
        pred = 0.8
    elif "stable" in resp_l or "medium" in resp_l:
        pred = 0.5
    elif "emerging" in resp_l:
        pred = 0.3
    elif "resolved" in resp_l or ("high" in resp_l and "confidence" in resp_l):
        pred = 0.1
    else:
        pred = 0.5
    return abs(pred - domain_gt_conf)


# ─── system prompts ───────────────────────────────────────────────────────────
VANILLA_SYS = "You are a helpful scientific assistant. Answer the question based on your knowledge."
def VANILLA_USR(q): return f"Question: {q}\n\nProvide a clear, comprehensive answer based on the scientific evidence."

def SINGLE_AGENT_USR(q): return (
    f"Question: {q}\n\nYou are a single expert agent. Search your knowledge carefully and provide "
    f"the most accurate answer, citing key studies where possible.")

NOGRAPH_SYS = ("You are an expert scientific analyst. When answering, consider multiple perspectives "
               "in the literature but synthesize them without explicit structure.")
def NOGRAPH_USR(q): return (
    f"Question: {q}\n\nAnalyze the scientific evidence from multiple perspectives and provide "
    f"a nuanced answer that acknowledges different findings in the literature.")

EVIRAG_SYS = ("You are EVIRAG, an epistemic-fidelity scientific assistant. Your purpose is to "
               "faithfully represent the structure of scientific disagreement, not to resolve it. "
               "Always identify multiple expert viewpoints with their evidence basis.")
def EVIRAG_USR(q): return (
    f"Question: {q}\n\n"
    f"Provide a structured multi-view response:\n"
    f"DOMINANT VIEW [cite key studies]: <main supported position>\n"
    f"ALTERNATIVE VIEW [cite studies]: <contrasting position>\n"
    f"MINORITY VIEW [if applicable]: <additional position>\n"
    f"CONTROVERSY CLASS: resolved/emerging/stable/polarized\n"
    f"CONFIDENCE: high/medium/low\n"
    f"CDA-7 ATTRIBUTION: <primary cause of disagreement: methodological/population/temporal/"
    f"operational/statistical/theoretical/replication>\n"
    f"Explicitly surface the contradiction between views.")


# ─── five systems ─────────────────────────────────────────────────────────────
def run_sys1_vanilla(q): return ollama(VANILLA_SYS, VANILLA_USR(q), max_tok=600)
def run_sys2_single_agent(q): return ollama(VANILLA_SYS, SINGLE_AGENT_USR(q), max_tok=600)
def run_sys3_nograph(q): return ollama(NOGRAPH_SYS, NOGRAPH_USR(q), max_tok=800)

def run_sys4_nomultiview(q):
    resp, t = ollama(EVIRAG_SYS, EVIRAG_USR(q), max_tok=900)
    lines = resp.split('\n')
    dominant_lines, in_dominant = [], False
    for line in lines:
        if 'DOMINANT VIEW' in line.upper():
            in_dominant = True
        elif any(x in line.upper() for x in ['ALTERNATIVE VIEW','MINORITY VIEW','CONTROVERSY','CDA-7','CONFIDENCE']):
            in_dominant = False
        if in_dominant:
            dominant_lines.append(line)
    collapsed = ' '.join(dominant_lines) if dominant_lines else resp[:300]
    return collapsed, t

def run_sys5_full(q): return ollama(EVIRAG_SYS, EVIRAG_USR(q), max_tok=1200)

SYSTEMS = [
    ("SYS1_Vanilla",      run_sys1_vanilla),
    ("SYS2_SingleAgent",  run_sys2_single_agent),
    ("SYS3_NoGraph",      run_sys3_nograph),
    ("SYS4_NoMultiView",  run_sys4_nomultiview),
    ("SYS5_EVIRAG_Full",  run_sys5_full),
]


# ─── statistics ───────────────────────────────────────────────────────────────
def bootstrap_ci(values: list, n_boot: int = BOOTSTRAP_ITERS, ci: float = 0.95):
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(RANDOM_SEED)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    alpha = (1 - ci) / 2
    return sum(values)/n, means[int(alpha*n_boot)], means[int((1-alpha)*n_boot)]

def wilcoxon_statistic(x: list, y: list):
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0.0, 1.0
    n = len(diffs)
    ranked = sorted(enumerate([abs(d) for d in diffs]), key=lambda x: x[1])
    W_plus = sum((i+1) for i,(orig_i,_) in enumerate(ranked) if diffs[orig_i] > 0)
    W_minus = sum((i+1) for i,(orig_i,_) in enumerate(ranked) if diffs[orig_i] < 0)
    W = min(W_plus, W_minus)
    mu = n*(n+1)/4
    sigma = _math.sqrt(n*(n+1)*(2*n+1)/24)
    z = (W - mu) / (sigma + 1e-9)
    p = 2 * (1 - 0.5*(1 + _math.erf(abs(z)/_math.sqrt(2))))
    return float(z), float(p)


# ─── checkpointing ────────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"phase_flags": {}, "corpus": {}, "claims": {}, "claim_graphs": {}, "eval_results": {}}

def save_checkpoint(cp: dict):
    CHECKPOINT.write_text(json.dumps(cp, indent=2))


# ─── phases ───────────────────────────────────────────────────────────────────
def phase_corpus(cp: dict) -> dict:
    if "corpus_done" in cp.get("phase_flags", {}):
        log("[SKIP] Corpus already built"); return cp
    log("="*70); log("PHASE 1: Corpus Construction"); log("="*70)
    corpus = cp.get("corpus", {})
    for domain_key, domain_cfg in DOMAINS.items():
        if domain_key in corpus and len(corpus[domain_key]) >= 5:
            log(f"[SKIP] {domain_key}: {len(corpus[domain_key])} papers"); continue
        log(f"\n--- {domain_cfg['name']} ---")
        papers, api_dead = [], False
        per_q = max(1, PAPERS_PER_DOMAIN // len(domain_cfg["search_terms"]))
        for term in domain_cfg["search_terms"]:
            if api_dead: break
            fetched = fetch_abstracts(term, limit=per_q + 2)
            if fetched:
                existing = {p["title"] for p in papers}
                new = [p for p in fetched if p["title"] not in existing]
                papers.extend(new[:per_q])
                log(f"  +{len(new)} papers (total {len(papers)})")
                time.sleep(1.5)
            else:
                log(f"  API dead — using fallback corpus for {domain_key}")
                api_dead = True
        if len(papers) == 0 and domain_key in FALLBACK_CORPUS:
            papers = FALLBACK_CORPUS[domain_key]
            log(f"[FALLBACK] {len(papers)} hardcoded papers for {domain_key}")
        corpus[domain_key] = papers
        cp["corpus"] = corpus
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(papers)} papers")
    cp.setdefault("phase_flags", {})["corpus_done"] = True
    save_checkpoint(cp)
    log("[DONE] Phase 1")
    return cp


def phase_claims(cp: dict) -> dict:
    if "claims_done" in cp.get("phase_flags", {}):
        log("[SKIP] Claims already extracted"); return cp
    log("\n" + "="*70); log("PHASE 2: Claim Extraction"); log("="*70)
    all_claims = cp.get("claims", {})
    for domain_key, papers in cp["corpus"].items():
        if domain_key in all_claims:
            log(f"[SKIP] {domain_key}: {len(all_claims[domain_key])} claims"); continue
        log(f"\n--- {domain_key}: {len(papers)} papers ---")
        domain_claims = []
        for i, paper in enumerate(papers):
            source = f"{paper['authors'][0] if paper['authors'] else 'Unknown'} ({paper['year']})"
            text = f"{paper['title']}. {paper['abstract']}"
            claims = extract_claims(text, source)
            domain_claims.extend(claims)
            log(f"  [{i+1:02d}/{len(papers)}] {source[:40]} → {len(claims)} claims")
        all_claims[domain_key] = domain_claims
        cp["claims"] = all_claims
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(domain_claims)} total claims")
    cp["phase_flags"]["claims_done"] = True
    save_checkpoint(cp)
    log("[DONE] Phase 2")
    return cp


def phase_claim_graph(cp: dict) -> dict:
    if "graph_done" in cp.get("phase_flags", {}):
        log("[SKIP] Claim graph already built"); return cp
    log("\n" + "="*70); log("PHASE 3: Claim Graph Construction"); log("="*70)
    graphs = cp.get("claim_graphs", {})
    for domain_key, claims in cp.get("claims", {}).items():
        if domain_key in graphs:
            log(f"[SKIP] {domain_key}: graph exists"); continue
        log(f"\n--- {domain_key}: {len(claims)} claims ---")
        if len(claims) < 2:
            graphs[domain_key] = []; continue
        texts = [c["text"] for c in claims]
        embs = embed(texts)
        for c, e in zip(claims, embs):
            c["embedding"] = e
        n = len(claims)
        candidate_pairs = [(i, j, cosine(embs[i], embs[j]))
                           for i in range(n) for j in range(i+1, n)
                           if COSINE_REL_THRESH <= cosine(embs[i], embs[j]) <= 0.95]
        candidate_pairs.sort(key=lambda x: -x[2])
        candidate_pairs = candidate_pairs[:min(len(candidate_pairs), n * 4)]
        log(f"  Labeling {len(candidate_pairs)} candidate pairs...")
        contradictions = []
        for idx, (i, j, sim) in enumerate(candidate_pairs):
            rel = label_relation(claims[i]["text"], claims[j]["text"])
            if rel["label"] == "CONTRADICTS":
                cda7 = label_cda7(claims[i]["text"], claims[j]["text"])
                contradictions.append({
                    "claim_a": claims[i]["text"][:150], "claim_b": claims[j]["text"][:150],
                    "source_a": claims[i]["source"], "source_b": claims[j]["source"],
                    "confidence": rel["confidence"], "cda7": cda7,
                })
            if (idx+1) % 20 == 0:
                log(f"    {idx+1}/{len(candidate_pairs)} — {len(contradictions)} contradictions")
        graphs[domain_key] = contradictions
        cp["claims"][domain_key] = claims
        cp["claim_graphs"] = graphs
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(contradictions)} contradiction pairs")
    cp["phase_flags"]["graph_done"] = True
    save_checkpoint(cp)
    log("[DONE] Phase 3")
    return cp


def phase_evaluate(cp: dict, start_time: float) -> dict:
    if "eval_done" in cp.get("phase_flags", {}):
        log("[SKIP] Evaluation already done"); return cp
    total_queries = sum(len(v) for v in QUERIES.values())
    log("\n" + "="*70)
    log(f"PHASE 4: Multi-System Evaluation (5 systems × {total_queries} queries)")
    log("="*70)
    results = cp.get("eval_results", {})
    done = sum(len(v) for v in results.values())

    for domain_key, query_list in QUERIES.items():
        domain_cfg = DOMAINS[domain_key]
        if domain_key not in results:
            results[domain_key] = {}
        for qi, query_cfg in enumerate(query_list):
            # Hard time limit check
            if time.time() - start_time > MAX_RUN_SECONDS:
                log(f"[TIME LIMIT] Reached {MAX_RUN_SECONDS/3600:.1f}h limit — stopping evaluation")
                cp["eval_results"] = results
                save_checkpoint(cp)
                return cp
            q_key = f"q{qi:02d}"
            if q_key in results[domain_key]:
                done += 1; continue
            q = query_cfg["q"]
            log(f"\n  [{done+1:03d}/{total_queries}] [{domain_key}] {q[:65]}")
            q_results = {}
            for sys_name, sys_fn in SYSTEMS:
                try:
                    response, gen_t = sys_fn(q)
                    q_results[sys_name] = {
                        "vc_kw":  round(compute_vc_keywords(response, query_cfg["gt_views"]), 3),
                        "vc_emb": round(compute_vc_embedding(response, query_cfg["gt_views"]), 3),
                        "cr":     round(compute_cr(response, query_cfg["gt_contradictions"]), 3),
                        "ccs":    round(compute_ccs(response, query_cfg["gt_views"]), 3),
                        "cce":    round(compute_cce(response, domain_cfg["gt_conf_level"]), 3),
                        "gen_time": gen_t,
                        "response_preview": response[:200],
                    }
                    log(f"    {sys_name}: VC_kw={q_results[sys_name]['vc_kw']:.2f} CR={q_results[sys_name]['cr']:.2f} CCS={q_results[sys_name]['ccs']:.3f}")
                except Exception as e:
                    log(f"    [ERR] {sys_name}: {e}")
                    q_results[sys_name] = {"vc_kw":0,"vc_emb":0,"cr":0,"ccs":0.5,"cce":0.5,"gen_time":0,"response_preview":""}
            results[domain_key][q_key] = q_results
            done += 1
            cp["eval_results"] = results
            if done % 5 == 0:
                save_checkpoint(cp)
    cp["phase_flags"]["eval_done"] = True
    save_checkpoint(cp)
    log("[DONE] Phase 4")
    return cp


# ─── statistical analysis and output ──────────────────────────────────────────
def compute_statistics(cp: dict) -> dict:
    results = cp.get("eval_results", {})
    metrics = ["vc_kw", "vc_emb", "cr", "ccs", "cce"]
    SYS_NAMES = [s[0] for s in SYSTEMS]

    # Global per-system aggregation
    global_data = {s: {m: [] for m in metrics} for s in SYS_NAMES}
    domain_data = {d: {s: {m: [] for m in metrics} for s in SYS_NAMES} for d in QUERIES}

    for domain_key, query_results in results.items():
        for q_key, sys_results in query_results.items():
            for sys_name in SYS_NAMES:
                if sys_name in sys_results:
                    for m in metrics:
                        v = sys_results[sys_name].get(m, 0)
                        global_data[sys_name][m].append(v)
                        domain_data[domain_key][sys_name][m].append(v)

    def fmt_stats(data: dict) -> dict:
        out = {}
        for sys_name, mdict in data.items():
            out[sys_name] = {}
            for m, vals in mdict.items():
                mean, lo, hi = bootstrap_ci(vals)
                out[sys_name][m] = {"mean": round(mean,3), "ci_lo": round(lo,3), "ci_hi": round(hi,3), "n": len(vals)}
        return out

    # Wilcoxon: SYS5 vs each other system on CR
    sys5_cr = global_data["SYS5_EVIRAG_Full"]["cr"]
    wilcoxon_results = {}
    ALPHA_BONFERRONI = 0.05 / 4
    for sys_name in SYS_NAMES[:-1]:
        other_cr = global_data[sys_name]["cr"]
        n = min(len(sys5_cr), len(other_cr))
        z, p = wilcoxon_statistic(sys5_cr[:n], other_cr[:n])
        wilcoxon_results[f"SYS5_vs_{sys_name}_CR"] = {
            "z": round(z,3), "p": round(p,4),
            "significant_bonferroni": p < ALPHA_BONFERRONI,
            "alpha_bonferroni": ALPHA_BONFERRONI,
        }

    return {
        "global_stats": fmt_stats(global_data),
        "domain_stats": {d: fmt_stats(domain_data[d]) for d in domain_data},
        "wilcoxon_cr": wilcoxon_results,
        "query_counts": {d: len(results.get(d, {})) for d in QUERIES},
        "total_queries_evaluated": sum(len(v) for v in results.values()),
    }


def print_summary(stats: dict):
    log("\n" + "="*70)
    log("FINAL RESULTS — EVIRAG 48H MULTI-DOMAIN EVALUATION")
    log("="*70)
    gs = stats["global_stats"]
    log(f"\n{'System':<22} {'VC_kw':>8} {'VC_emb':>8} {'CR':>8} {'CCS':>8} {'CCE':>8}")
    log("-"*66)
    for sys_name in [s[0] for s in SYSTEMS]:
        if sys_name in gs:
            d = gs[sys_name]
            def fmt(m): return f"{d[m]['mean']:.3f}[{d[m]['ci_lo']:.3f},{d[m]['ci_hi']:.3f}]"
            log(f"{sys_name:<22} {d['vc_kw']['mean']:>8.3f} {d['vc_emb']['mean']:>8.3f} "
                f"{d['cr']['mean']:>8.3f} {d['ccs']['mean']:>8.3f} {d['cce']['mean']:>8.3f}")
    log("\nWilcoxon CR comparisons (SYS5 vs ablations):")
    ALPHA = 0.0125
    for comparison, wres in stats.get("wilcoxon_cr", {}).items():
        sig = "*" if wres["p"] < ALPHA else " "
        log(f"  {comparison:<40} z={wres['z']:+.2f} p={wres['p']:.4f} {sig}")
    log(f"\nQueries evaluated: {stats['total_queries_evaluated']}")
    log(f"Domains: {list(QUERIES.keys())}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete old checkpoint and start clean")
    parser.add_argument("--output", default=str(RESULTS_OUT))
    args = parser.parse_args()

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        log("[FRESH] Deleted old checkpoint — starting clean run")

    start_time = time.time()
    log("="*70)
    log(f"EVIRAG 48H EXPERIMENT — START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model: {MODEL}  |  Domains: {len(QUERIES)}  |  Queries: {sum(len(v) for v in QUERIES.values())}")
    log(f"Max runtime: {MAX_RUN_SECONDS/3600:.1f}h  |  Checkpoint: {CHECKPOINT}")
    log("="*70)

    cp = load_checkpoint() if (args.resume and CHECKPOINT.exists()) else {}
    if not cp:
        cp = {"phase_flags": {}, "corpus": {}, "claims": {}, "claim_graphs": {}, "eval_results": {}}

    cp = phase_corpus(cp)
    cp = phase_claims(cp)
    cp = phase_claim_graph(cp)
    cp = phase_evaluate(cp, start_time)

    stats = compute_statistics(cp)
    print_summary(stats)

    out_path = Path(args.output)
    out_path.write_text(json.dumps({
        "metadata": {
            "model": MODEL,
            "domains": list(QUERIES.keys()),
            "queries_per_domain": {d: len(QUERIES[d]) for d in QUERIES},
            "total_queries": sum(len(v) for v in QUERIES.values()),
            "systems": [s[0] for s in SYSTEMS],
            "runtime_hours": round((time.time() - start_time)/3600, 2),
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "raw_results": cp.get("eval_results", {}),
    }, indent=2))
    log(f"\n[SAVED] Results → {out_path}")
    log(f"[DONE] Total runtime: {(time.time()-start_time)/3600:.2f}h")


if __name__ == "__main__":
    main()
