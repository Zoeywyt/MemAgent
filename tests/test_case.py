import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.empathy_agent import EmpathyAgent
from output_store import (
    consulting_report_checkpoint_path,
    consulting_report_json_path,
    consulting_report_log_path,
    ensure_output_dirs,
    load_user_session_summaries,
    load_user_trends,
    record_session_outputs,
)


CASE_DATA: Dict[str, Any] = {
    "user_id": "CP224_刘某",
    "sessions": [
        {
            "session_id": "Session1",
            "messages": [
                {"role": "user", "content": "来访者：（叹气，低着头）医生，我也不知道怎么说，就是觉得活着特别没意思，干什么都提不起劲。"},
                {"role": "assistant", "content": "[开放式提问] 这种“没意思”的感觉是从什么时候开始的？你能具体说说，在日常生活里它是怎么影响你的吗？"},
                {"role": "user", "content": "来访者：差不多一年了。以前还好，最近几个月越来越重。饭吃不下，觉也睡不好，整天昏昏沉沉的，看书根本看不进去。"},
                {"role": "assistant", "content": "[共情与具体化] 听起来你的身体和精神都在承受很大的负担。特别是看书这一点，能详细说说吗？"},
                {"role": "user", "content": "来访者：就是……我根本集中不了注意力。尤其是快考试的时候，我头就疼，心慌得厉害，就像要大难临头一样。晚上躺床上翻来覆去，越想睡越睡不着。"},
                {"role": "assistant", "content": "[情绪聚焦] “大难临头”，这个词你用得很重。是怕考试本身，还是怕考试带来的后果？"},
                {"role": "user", "content": "来访者：都怕。其实我最后也都能及格，但我就是觉得自己考得太差了，没用，对不起我爸妈。他们花那么多钱供我读这个书……"},
                {"role": "assistant", "content": "[情感反映] 所以成绩及格并不能让你松一口气，反而让你更自责。你的标准似乎比及格线要高得多。"},
                {"role": "user", "content": "来访者：我本来就不该只是及格！我……我就是太没用了。别人都能学好，就我不行。"},
                {"role": "assistant", "content": "[探索自我评价] 你说“我不行”，这个评价是怎么来的？是成绩单告诉你的，还是你自己心里一直有这样一个声音？"},
                {"role": "user", "content": "来访者：（沉默片刻）可能是我自己觉得吧。我从小就不如别人，什么都得听家里的。"},
                {"role": "assistant", "content": "[引导联系过去] 听家里的？能多谈谈你小时候和父母相处的模式吗？"},
                {"role": "user", "content": "来访者：我爸妈说什么就是什么，我从来不顶嘴。他们觉得好的，我就去做；觉得不好的，我就不能碰。高考没考上，他们托人找了份工作，我干着也挺好，可他们后来又觉得不好，硬是给我辞了，让我来自费读这个热门专业。"},
                {"role": "assistant", "content": "[澄清与共情] 所以，辞掉你喜欢的工作，来读这个专业，完全是父母的决定。你当时心里是怎么想的？"},
                {"role": "user", "content": "来访者：我能怎么想？他们是为我好，这个专业将来好找工作，有出息。我不能辜负他们。"},
                {"role": "assistant", "content": "[面质与引导] “我不能辜负他们”，这句话里好像没有你自己的意愿。你有没有想过，你自己真正想做什么？"},
                {"role": "user", "content": "来访者：（抬头看了一眼医生，又低下）没……没想过。想了也没用，反正也改变不了什么。"},
                {"role": "assistant", "content": "[共情] 我能感觉到一种深深的无力感。就好像你的人生方向盘一直握在别人手里，而你只能坐在副驾驶上，心里再着急也使不上劲。"},
                {"role": "user", "content": "来访者：（眼眶微红）对，就是这种感觉。我特别羡慕那些能自己做主的人。我现在一进教室就烦，那些专业课我一点兴趣都没有，都是硬着头皮在学。"},
                {"role": "assistant", "content": "[命名冲突] 一边是内心极度的厌倦和反感，另一边是“必须学好、必须让父母满意”的巨大压力。你每天就被夹在这两种力量中间，对吗？"},
                {"role": "user", "content": "来访者：对！所以我头痛、失眠、心慌。我甚至盼着生病，生病了就有理由不去上课了。可是真的缺了课，我又恨自己，觉得自己在逃避，更没用了。"},
                {"role": "assistant", "content": "[解释症状的意义] 你看，症状在这里成了一个矛盾的“解决方案”。头痛让你暂时逃离了让你痛苦的课堂，但随之而来的自责又让你陷入更深的痛苦。这是一个恶性循环。"},
                {"role": "user", "content": "来访者：那我该怎么办？我总不能一直这么头痛下去吧？"},
                {"role": "assistant", "content": "[设定初步目标] 我们今天先不急着找“怎么办”的答案，而是先看清“是什么”让你如此痛苦。你今天的表达非常清晰，我们找到了一个很重要的线索——你的意愿和你父母意愿之间的冲突。下次我们可以继续沿着这个方向，去看看这种“顺从”的模式是如何形成的。"},
                {"role": "user", "content": "来访者：好吧，医生，跟您说完，心里好像松快了一点点。"},
                {"role": "assistant", "content": "[鼓励] 这是一个好的开始。允许自己把这些压抑已久的话说出来，本身就是一种疗愈。我们下周再见。"},
            ],
        },
        {
            "session_id": "Session2",
            "messages": [
                {"role": "user", "content": "来访者：医生，这周我还是老样子，一想到要考试了，就整晚整晚睡不着。"},
                {"role": "assistant", "content": "[连接上次内容] 上次我们谈到，你的身体在用症状表达内心的冲突。这次考试焦虑出现时，你脑子里最先冒出来的念头是什么？"},
                {"role": "user", "content": "来访者：就是怕考不好。怕看到成绩单上那个刚及格的分，怕我爸叹气，怕我妈说“我们为你付出这么多，你就拿这个回报我们？”"},
                {"role": "assistant", "content": "[具体化威胁] 你描绘了一个非常清晰的画面。这个画面里，你父母是审判者，而你是一个等待宣判的罪人。成绩单就是判决书。"},
                {"role": "user", "content": "来访者：您这么一说，还真是。我特别怕让他们失望。"},
                {"role": "assistant", "content": "[探索深层动机] 除了怕他们失望，你心底里有没有一丝丝，哪怕一丁点，对他们替你做的这个选择的……不满？"},
                {"role": "user", "content": "来访者：（急忙摇头）没有没有，他们是为我好。我怎么能怪他们呢？"},
                {"role": "assistant", "content": "[正常化与接纳] 有不满是很正常的情绪，这不等于你不孝顺，也不等于你否定他们对你的爱。试着去感受一下，把“他们是为我好”这个道理先放一边，你心里有没有委屈？"},
                {"role": "user", "content": "来访者：（沉默很久，声音变得很低）委屈……有。我有时候会想，凭什么他们觉得不好就给我辞了？我干得挺开心的。现在学的这个，我根本就不喜欢，每天上课跟上刑一样。"},
                {"role": "assistant", "content": "[共情与肯定] 你能说出“委屈”和“上刑”这两个词，非常不容易。这说明你开始允许自己接触真实的情感了。这就是你的“自我”在发出声音。"},
                {"role": "user", "content": "来访者：可是……有这种想法是不是很自私？他们都是为了我的前途。"},
                {"role": "assistant", "content": "[认知重构] 为自己的感受和意愿发声，这不是自私，这是心理上“成年”的标志。一个健康的人，是先能为自己负责，然后才能为别人负责。你现在把所有力气都用来为父母的期待负责，你还有力气为自己负责吗？"},
                {"role": "user", "content": "来访者：我……我好像从来没从这个角度想过。我一直觉得听他们的话就是对的。"},
                {"role": "assistant", "content": "[探索成长史] 这种“听话就是对的”模式，是从小养成的吧？"},
                {"role": "user", "content": "来访者：嗯。我从小就很乖，邻居都夸我懂事。我爸妈说什么我都听，不让他们操心。"},
                {"role": "assistant", "content": "[解释代价] “懂事”和“听话”让你获得了安全和赞赏，但代价是你压抑了自己真实的想法和感受。久而久之，你可能连自己想要什么都不知道了。现在这种迷茫和痛苦，就是被压抑的部分在向你抗议。"},
                {"role": "user", "content": "来访者：您说的我有点明白了。我确实不知道自己想要什么，好像除了让父母满意，我活着就没别的目标了。"},
                {"role": "assistant", "content": "[鼓励自我探索] 没关系，不知道是正常的，因为你的“自我”才刚刚被允许探出头来。我们可以慢慢来。你回忆一下，在人生中，有没有那么一件事，是你完全凭自己兴趣去做，并且感到快乐的？哪怕是很小的事。"},
                {"role": "user", "content": "来访者：（想了很久）……我上中专的时候，偷偷学过一段时间吉他，没敢跟家里说，怕他们说我不务正业。那时候弹琴的时候，我确实挺开心的。"},
                {"role": "assistant", "content": "[强化积极体验] 吉他。这是一个非常重要的线索。它证明了你是有自己的兴趣和热情的，只是它一直被“不务正业”的评判给压制了。记住弹吉他时那种由内而外的快乐感觉，那就是“你的意愿”的样子。"},
                {"role": "user", "content": "来访者：可是学吉他也不能当饭吃啊。"},
                {"role": "assistant", "content": "[区分问题] 我们现在探讨的不是职业规划，而是寻找你内心的火种。一个连自己内心的火都感受不到的人，是不可能找到真正适合自己的人生道路的。我们先学习聆听自己，好吗？"},
                {"role": "user", "content": "来访者：好。我试试。"},
                {"role": "assistant", "content": "[布置家庭作业] 这周回去，请你做一个小练习：每天找一个安静的时刻，问问自己“此刻，不考虑任何人的看法，我想做什么？”哪怕答案只是“我想发呆”，也请尊重它。我们下次来聊聊你的答案。"},
                {"role": "user", "content": "来访者：好的，医生，这对我来说有点难，但我愿意试试。"},
                {"role": "assistant", "content": "[鼓励] 迈出这一步，本身就是巨大的勇气。我们下周见。"},
            ],
        },
        {
            "session_id": "Session3",
            "messages": [
                {"role": "user", "content": "来访者：医生，我上周试着做了那个练习。"},
                {"role": "assistant", "content": "[开放式提问] 哦？感觉怎么样？你得到了哪些答案？"},
                {"role": "user", "content": "来访者：答案很奇怪。第一天我想发呆，第二天我想去公园走走，第三天我居然想去吃一顿好的。我以前从来不会想这些，觉得都是浪费时间。"},
                {"role": "assistant", "content": "[肯定与解释] 这些一点都不奇怪，这些都是一个正常人最基本的心理需求。你觉得奇怪，恰恰说明过去你把“满足自己”和“浪费时间”划上了等号。"},
                {"role": "user", "content": "来访者：我还发现，当我真的去公园走了一圈之后，那天晚上我睡得特别好，也没头痛。"},
                {"role": "assistant", "content": "[连接行为与感受] 这是一个非常重要的发现。当你遵从内心的微小愿望时，身体的警报就暂时解除了。这证明了你的症状和内心冲突的直接关联。"},
                {"role": "user", "content": "来访者：可是，这种轻松感持续不了多久。我一回到学校，看到课本，那种烦躁和压力就又回来了。"},
                {"role": "assistant", "content": "[直面核心冲突] 因为根本的矛盾还没有解决。课本就象征着父母强加给你的意愿，你只要还坐在那个教室里，冲突就一直在。"},
                {"role": "user", "content": "来访者：那我该怎么办？我总不能真的退学吧？那不全完了？我爸妈非气死不可。"},
                {"role": "assistant", "content": "[探索可能性] 我们来把问题拆开看。你怕的不是“退学”这个行为，而是“退学后父母的反应和你无法承受的后果”。我们先不想退不退学，我们先想，有没有一条路，是既能照顾到你的感受，又能有一个让父母相对能接受的方案？"},
                {"role": "user", "content": "来访者：什么方案？换专业？"},
                {"role": "assistant", "content": "[引导自我负责] 这是一个选项。但关键在于，换什么？你不能从一个不喜欢的专业，换到另一个父母喜欢的专业。这个决定，必须是你自己经过探索后做出的。"},
                {"role": "user", "content": "来访者：我……我不知道自己喜欢什么专业。"},
                {"role": "assistant", "content": "[鼓励探索与尝试] 这不怪你。因为过去二十三年，你的喜好系统一直处于关闭状态。现在我们试着打开它。除了弹吉他，你平时还会关注些什么？看什么类型的视频？读什么书？"},
                {"role": "user", "content": "来访者：我……我喜欢看一些修理东西的视频，修手机、修电脑什么的，感觉把一个坏的东西弄好，挺有成就感的。"},
                {"role": "assistant", "content": "[发现优势资源] 这是一个非常具体且现实的兴趣！动手操作，解决实际问题，并获得即时反馈。这和坐在教室里学抽象的理论是完全不同的。你有没有想过，这可能就是你的天赋所在？"},
                {"role": "user", "content": "来访者：天赋？这能算什么天赋，不就是修东西吗。我爸妈肯定觉得这是不务正业。"},
                {"role": "assistant", "content": "[认知挑战] 我们的社会需要各种人才，一个优秀的维修工程师收入和社会地位不一定比白领低。问题的核心是，你能不能在心里先肯定自己的这个兴趣，赋予它价值，而不是先代入父母的眼光来否定它。"},
                {"role": "user", "content": "来访者：（若有所思）……"},
                {"role": "assistant", "content": "[探讨独立之路] 换一个思路。即使不换专业，我们是否可以先允许自己“及格就好”？把以前用来焦虑和内耗的精力，抽出一部分去发展你修理东西的兴趣？当你有了一个属于自己的、能带来成就感的角落，你对学业的负面情绪就不会占据你的全部生活了。"},
                {"role": "user", "content": "来访者：您的意思是，我可以一边混着这个文凭，一边去干我喜欢的事？"},
                {"role": "assistant", "content": "[肯定与支持] “混文凭”这个词，在你过去的字典里是不是大逆不道？但对你目前的情况来说，这恰恰是一种策略性的自我保护。你不需要在每一件事上都做到完美，你需要的是保存心理能量，为你真正的人生目标做准备。"},
                {"role": "user", "content": "来访者：我好像有点明白了。我以前觉得世界非黑即白，要么就拼命学好，要么就干脆不学。从来没想过还有中间这条路。"},
                {"role": "assistant", "content": "[总结] 是的，这就是心理的灵活性。我们今天的谈话，让你看到了第三种可能性。你不需要立刻做一个翻天覆地的决定，你需要的是调整心态，把生活的重心，慢慢从“让父母满意”挪回到“让自己舒服”上来。"},
                {"role": "user", "content": "来访者：我觉得心里那个大石头好像小了一点。"},
                {"role": "assistant", "content": "[展望] 这是一个很好的迹象。下次我们可以具体讨论，如何在你目前的环境里，为自己搭建一个修理东西的小天地。"},
            ],
        },
        {
            "session_id": "Session4",
            "messages": [
                {"role": "user", "content": "来访者：医生，这周我过得挺神奇的。"},
                {"role": "assistant", "content": "[开放式提问] “神奇”？听起来有一些不一样的事情发生了。"},
                {"role": "user", "content": "来访者：我上周回去，真就找了个废旧的小收音机拆着玩。结果捣鼓了两天，还真让我给弄响了。那个瞬间，我特别特别开心，好久没那么开心过了。"},
                {"role": "assistant", "content": "[强化积极体验] 这种通过自己动手解决问题而获得的快乐，是任何外部评价都给不了的。它是真正属于你的、坚固的快乐。"},
                {"role": "user", "content": "来访者：对！而且我发现，当我专心修东西的时候，什么烦恼都没了，也不想考试的事了。晚上睡得也踏实。"},
                {"role": "assistant", "content": "[解释疗愈机制] 这就是“心流”体验。当你沉浸在喜欢的事情里，你的心灵得到了休息，冲突暂时退场了。你在无意中找到了疗愈自己的一剂良药。"},
                {"role": "user", "content": "来访者：还有一件事。这周有个小测验，我还是紧张，但考的时候我脑子里冒出一个念头：“考及格就行了，反正我以后也不靠这个吃饭。” 然后我居然就没那么慌了。"},
                {"role": "assistant", "content": "[肯定认知转变] 这是一个里程碑式的转变！你成功地在考试场景中运用了新的认知——“及格就好”。它卸下了你追求完美的枷锁。"},
                {"role": "user", "content": "来访者：是啊，虽然最后成绩还是刚及格，但我心里没那么难受了。"},
                {"role": "assistant", "content": "[巩固成果] 你不再用父母的尺子来鞭打自己了。你开始拥有属于自己的、哪怕是很小的一把尺子。"},
                {"role": "user", "content": "来访者：医生，我最近在想，等这学期结束，我想去找个修手机的店当学徒试试，哪怕不要工资也行。我想看看我是不是真的喜欢干这个。"},
                {"role": "assistant", "content": "[鼓励独立探索] 这是一个非常勇敢的想法。从“想想”到“试试”，这是最关键的一步。你不需要先告诉父母，可以先去体验。把这件事当作你送给自己的一个秘密礼物。"},
                {"role": "user", "content": "来访者：可是……我爸妈那边，我终究还是要说的吧？"},
                {"role": "assistant", "content": "[讨论应对策略] 是的，但时机很重要。当你只是说“我想去”，父母看到的是你的“任性”。但当你有了实践经验，甚至能拿出一份学徒计划时，你向他们展示的是你的“认真”和“规划”。前者引发冲突，后者争取对话。"},
                {"role": "user", "content": "来访者：我懂了。我要先让自己有底气，才能去跟他们谈。"},
                {"role": "assistant", "content": "[强化自我意识] 没错。你正在为自己的人生积蓄底气。我们咨询的目标，不是帮你解决要不要退学这一个问题，而是帮你长出能够应对人生各种选择的能力——独立思考、尊重自我、承担责任。"},
                {"role": "user", "content": "来访者：我以前觉得我的人生是条死胡同，现在好像看到前面有个拐角，虽然不知道拐过去是什么，但我有想走过去看看的冲动了。"},
                {"role": "assistant", "content": "[共情与展望] 从“大难临头”到“想去拐角看看”，这个转变非常了不起。你对生活的希望感正在复苏。"},
                {"role": "user", "content": "来访者：谢谢您，医生。"},
                {"role": "assistant", "content": "[结束语与鼓励] 最应该感谢的是你自己，是你自己鼓起勇气走进了咨询室，并坚持探索。我们的咨询可以告一段落，但你的自我成长之路才刚刚开始。记住，当你感到困惑时，随时可以回来。祝你顺利。"},
            ],
        },
    ],
}
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def truncate_text(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def summarize_l1(l1_summary: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(l1_summary, dict):
        return {}
    return {
        "主题": truncate_text(l1_summary.get("主题") or l1_summary.get("topic")),
        "背景": truncate_text(l1_summary.get("背景") or l1_summary.get("background")),
        "会话总结": truncate_text(l1_summary.get("会话总结") or l1_summary.get("summary")),
    }


def summarize_retrieval_for_log(retrieval: Dict[str, Any], limit: int = 3) -> Dict[str, Any]:
    l3_fragments = retrieval.get("l3_fragments", []) or []
    graph_relations = retrieval.get("graph_relations", []) or []
    return {
        "l3_count": len(l3_fragments),
        "graph_relation_count": len(graph_relations),
        "l3_top_hits": [truncate_text(item.get("memory", ""), 100) for item in l3_fragments[:limit]],
        "graph_relation_top_hits": [
            f"{item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
            for item in graph_relations[:limit]
        ],
        "context_preview": truncate_text(
            "\n".join(
                [item.get("memory", "") for item in l3_fragments[:limit]]
                + [
                    f"{item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
                    for item in graph_relations[:limit]
                ]
            ),
            280,
        ),
    }


def summarize_l3_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stored": result.get("stored", 0),
        "facts": result.get("facts", []),
        "actions": result.get("actions", []),
    }


def summarize_preloaded_context(session_info: Dict[str, Any]) -> Dict[str, Any]:
    treatment_report = session_info.get("treatment_report") or {}
    return {
        "l1_summary": summarize_l1(session_info.get("l1_summary")),
        "l2_summary_count": len(session_info.get("l2_summaries") or []),
        "l2_top_topics": [
            item.get("topic", "")
            for item in (session_info.get("l2_summaries") or [])[:3]
        ],
        "has_treatment_report": bool(treatment_report),
        "treatment_phase": treatment_report.get("treatment_phase", ""),
        "next_focus": truncate_text(treatment_report.get("next_focus", ""), 180),
        "system_prompt_context_preview": truncate_text(session_info.get("preloaded_context_text", ""), 360),
    }


def summarize_graph_relations(graph_data: Dict[str, Any], limit: int = 10) -> List[str]:
    relations = graph_data.get("relations", []) or []
    return [
        f"{item.get('source', '')} --{item.get('relation', '')}--> {item.get('destination', '')}"
        for item in relations[:limit]
    ]


def to_clickable_uri(path: Path) -> str:
    return path.resolve().as_uri()


class ReplayLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def log(self, title: str, payload: Optional[Dict[str, Any]] = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{timestamp}] {title}"]
        if payload is not None:
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        text = "\n".join(lines)
        print(text)
        print("-" * 80)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n" + "-" * 80 + "\n")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(
    checkpoint_path: Path,
    *,
    user_id: str,
    status: str,
    session_results: List[Dict[str, Any]],
    completed_sessions: List[str],
    last_committed_l1_summary: Optional[Dict[str, Any]],
    current_session: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    write_json(
        checkpoint_path,
        {
            "user_id": user_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "current_session": current_session,
            "completed_sessions": completed_sessions,
            "last_committed_l1_summary": last_committed_l1_summary,
            "last_error": last_error,
            "session_results": session_results,
        },
    )


def restore_l1_snapshot(agent: EmpathyAgent, user_id: str, l1_summary: Optional[Dict[str, Any]]) -> None:
    summary_store = agent.summary_agent.summary_store
    summary_path = summary_store.storage_dir / f"{user_id}_summary.json"

    if l1_summary is None:
        if summary_path.exists():
            summary_path.unlink()
        return

    summary_store.save_user_summary(user_id, l1_summary)


def reset_partial_session(agent: EmpathyAgent, user_id: str, session_label: str) -> None:
    run_session_id = f"{user_id}_{session_label}"
    try:
        agent.mem0.client.delete_all(user_id=user_id, run_id=run_session_id)
        print(f"已清理未完成会话残留：{run_session_id}")
    except Exception as exc:
        print(f"清理未完成会话残留失败（可忽略，后续继续重跑）：{run_session_id} -> {exc}")


def build_result_bundle(
    agent: EmpathyAgent,
    user_id: str,
    session_results: List[Dict[str, Any]],
    completed_sessions: List[str],
    status: str,
    include_retrieval_examples: bool,
    last_error: Optional[str] = None,
) -> Dict[str, Any]:
    summaries_doc = load_user_session_summaries(user_id)
    trends_doc = load_user_trends(user_id)
    reports = trends_doc.get("reports", []) if isinstance(trends_doc.get("reports", []), list) else []
    latest_treatment_report: Dict[str, Any] = {}
    if reports:
        latest_treatment_report = max(
            (item for item in reports if isinstance(item, dict)),
            key=lambda item: str(item.get("generated_at", "")),
            default={},
        ).get("report", {})
    bundle: Dict[str, Any] = {
        "user_id": user_id,
        "status": status,
        "completed_sessions": completed_sessions,
        "session_results": session_results,
        "final_l1_summary": agent.summary_agent.get_user_summary(user_id),
        "all_l2_summaries": summaries_doc.get("sessions", [])[:20],
        "latest_treatment_report": latest_treatment_report,
        "last_error": last_error,
    }

    if include_retrieval_examples:
        bundle["retrieval_examples"] = {
            "考试焦虑与父母期待": {},
            "顺从模式与自我压抑": {},
            "兴趣探索与修理方向": {},
        }

    return bundle


def replay_session(
    agent: EmpathyAgent,
    user_id: str,
    session_label: str,
    messages: List[Dict[str, str]],
    replay_logger: Optional[ReplayLogger] = None,
) -> Dict[str, Any]:
    session_info = agent.start_session(user_id)
    run_session_id = f"{user_id}_{session_label}"
    agent.session_id = run_session_id

    l3_records: List[Dict[str, Any]] = []
    if replay_logger:
        replay_logger.log(
            f"{session_label} | session_start",
            {
                "run_session_id": run_session_id,
                "preloaded_session_context": summarize_preloaded_context(session_info),
            },
        )

    for index in range(0, len(messages), 2):
        user_message = messages[index] if index < len(messages) else None
        assistant_message = messages[index + 1] if index + 1 < len(messages) else None
        if not user_message or user_message.get("role") != "user":
            continue

        user_text = user_message.get("content", "")
        assistant_text = assistant_message.get("content", "") if assistant_message else ""
        turn_index = len(agent.session_messages) // 2 + 1

        if replay_logger:
            retrieval_result = agent.mem0.search_relevant_context(user_id=user_id, query=user_text, limit=3)
            replay_logger.log(
                f"{session_label} | turn_{turn_index} | retrieval_before_response",
                {
                    "query": truncate_text(user_text, 160),
                    "retrieval": summarize_retrieval_for_log(retrieval_result),
                    "files_written_so_far": {
                        "l1_summary_file": str(agent.summary_agent.summary_store.storage_dir / f"{user_id}_summary.json"),
                    },
                },
            )

        agent.session_messages.append({"role": "user", "content": user_text})
        if assistant_text:
            agent.session_messages.append({"role": "assistant", "content": assistant_text})

        l3_result = agent.mem0.remember_l3_turn(
            user_id=user_id,
            session_id=run_session_id,
            turn_index=len(agent.session_messages) // 2,
            user_input=user_text,
            assistant_response=assistant_text,
        )
        if replay_logger:
            replay_logger.log(
                f"{session_label} | turn_{turn_index} | l3_write_decision",
                {
                    "user": truncate_text(user_text, 160),
                    "assistant": truncate_text(assistant_text, 160),
                    "l3_memory_result": summarize_l3_result(l3_result),
                },
            )

        l3_records.append(
            {
                "turn_index": len(agent.session_messages) // 2,
                "user": user_text,
                "assistant": assistant_text,
                "l3_memory_result": l3_result,
            }
        )

    end_result = agent.end_session()
    output_paths = record_session_outputs(
        user_id=user_id,
        display_name=user_id,
        run_session_id=run_session_id,
        chat_pairs=[
            {"user": record.get("user", ""), "assistant": record.get("assistant", "")}
            for record in l3_records
        ],
        l2_summary=end_result.get("l2_summary", {}),
        treatment_report=end_result.get("treatment_report", {}),
        l1_summary=end_result.get("l1_summary", {}),
        archived_at=datetime.now().isoformat(),
        logical_session_id=session_label,
    )
    latest_l2 = agent.mem0.get_l2_summaries(user_id, limit=1)
    if replay_logger:
        replay_logger.log(
            f"{session_label} | session_end",
            {
                "latest_l2_summary": latest_l2[0] if latest_l2 else {},
                "updated_l1_summary": summarize_l1(end_result.get("l1_summary")),
                "session_graph_relations": summarize_graph_relations(end_result.get("graph_data", {})),
                "treatment_report": end_result.get("treatment_report", {}),
                "split_output_files": output_paths,
            },
        )

    return {
        "session_label": session_label,
        "run_session_id": run_session_id,
        "initial_l1_summary": session_info.get("l1_summary"),
        "preloaded_session_context": {
            "l1_summary": session_info.get("l1_summary"),
            "l2_summaries": session_info.get("l2_summaries", []),
            "treatment_report": session_info.get("treatment_report"),
            "preloaded_context_text": session_info.get("preloaded_context_text", ""),
        },
        "l3_records": l3_records,
        "end_result": end_result,
        "split_output_files": output_paths,
    }


def main() -> None:
    user_id = os.getenv("CASE_USER_ID", CASE_DATA["user_id"])
    ensure_output_dirs()
    output_path = consulting_report_json_path(user_id, prefix="replay_results")
    checkpoint_path = consulting_report_checkpoint_path(user_id)
    runtime_log_path = consulting_report_log_path(user_id)
    force_restart = os.getenv("REPLAY_FORCE_RESTART", "").strip().lower() in {"1", "true", "yes"}

    agent = EmpathyAgent()
    replay_logger = ReplayLogger(runtime_log_path)
    checkpoint = {} if force_restart else load_checkpoint(checkpoint_path)

    session_results: List[Dict[str, Any]] = checkpoint.get("session_results", [])
    completed_sessions: List[str] = checkpoint.get("completed_sessions", [])
    completed_session_set = set(completed_sessions)
    last_committed_l1_summary = checkpoint.get("last_committed_l1_summary")

    restore_l1_snapshot(agent, user_id, last_committed_l1_summary)
    replay_logger.log(
        "replay_start",
        {
            "user_id": user_id,
            "output_path": str(output_path),
            "checkpoint_path": str(checkpoint_path),
            "runtime_log_path": str(runtime_log_path),
            "log_focus": [
                "会话启动时自动预加载 L1 长期画像、L2 历史会话摘要、督导师治疗进展报告",
                "每轮检索前的 L3/graph_relations 调取",
                "L3 写入判定",
                "会话结束后的 L2/L1/graph/treatment_report",
                "checkpoint 与结果文件路径"
            ],
        },
    )

    for session in CASE_DATA["sessions"]:
        session_label = session["session_id"]
        if session_label in completed_session_set:
            replay_logger.log(f"{session_label} | skipped_from_checkpoint")
            continue

        replay_logger.log(
            f"{session_label} | processing",
            {
                "checkpoint_path": str(checkpoint_path),
                "output_path": str(output_path),
            },
        )
        restore_l1_snapshot(agent, user_id, last_committed_l1_summary)
        reset_partial_session(agent, user_id, session_label)
        save_checkpoint(
            checkpoint_path,
            user_id=user_id,
            status="running",
            session_results=session_results,
            completed_sessions=completed_sessions,
            last_committed_l1_summary=last_committed_l1_summary,
            current_session=session_label,
            last_error=None,
        )
        replay_logger.log(
            f"{session_label} | checkpoint_saved",
            {
                "current_session": session_label,
                "completed_sessions": completed_sessions,
            },
        )

        try:
            session_result = replay_session(
                agent=agent,
                user_id=user_id,
                session_label=session_label,
                messages=session["messages"],
                replay_logger=replay_logger,
            )
        except Exception as exc:
            save_checkpoint(
                checkpoint_path,
                user_id=user_id,
                status="failed",
                session_results=session_results,
                completed_sessions=completed_sessions,
                last_committed_l1_summary=last_committed_l1_summary,
                current_session=session_label,
                last_error=str(exc),
            )
            partial_bundle = build_result_bundle(
                agent=agent,
                user_id=user_id,
                session_results=session_results,
                completed_sessions=completed_sessions,
                status="failed",
                include_retrieval_examples=False,
                last_error=str(exc),
            )
            write_json(output_path, partial_bundle)
            replay_logger.log(
                f"{session_label} | failed",
                {
                    "error": str(exc),
                    "checkpoint_path": str(checkpoint_path),
                    "output_path": str(output_path),
                },
            )
            raise

        session_results.append(session_result)
        completed_sessions.append(session_label)
        completed_session_set.add(session_label)
        last_committed_l1_summary = agent.summary_agent.get_user_summary(user_id)

        save_checkpoint(
            checkpoint_path,
            user_id=user_id,
            status="running",
            session_results=session_results,
            completed_sessions=completed_sessions,
            last_committed_l1_summary=last_committed_l1_summary,
            current_session=None,
            last_error=None,
        )
        write_json(
            output_path,
            build_result_bundle(
                agent=agent,
                user_id=user_id,
                session_results=session_results,
                completed_sessions=completed_sessions,
                status="running",
                include_retrieval_examples=False,
            ),
        )
        replay_logger.log(
            f"{session_label} | completed",
            {
                "completed_sessions": completed_sessions,
                "latest_l1_summary": summarize_l1(last_committed_l1_summary),
                "output_path": str(output_path),
            },
        )

    save_checkpoint(
        checkpoint_path,
        user_id=user_id,
        status="finalizing",
        session_results=session_results,
        completed_sessions=completed_sessions,
        last_committed_l1_summary=last_committed_l1_summary,
        current_session="__finalize__",
        last_error=None,
    )

    result_bundle = build_result_bundle(
        agent=agent,
        user_id=user_id,
        session_results=session_results,
        completed_sessions=completed_sessions,
        status="completed",
        include_retrieval_examples=True,
    )
    write_json(output_path, result_bundle)
    save_checkpoint(
        checkpoint_path,
        user_id=user_id,
        status="completed",
        session_results=session_results,
        completed_sessions=completed_sessions,
        last_committed_l1_summary=last_committed_l1_summary,
        current_session=None,
        last_error=None,
    )
    replay_logger.log(
        "replay_completed",
        {
            "output_path": str(output_path),
            "checkpoint_path": str(checkpoint_path),
            "runtime_log_path": str(runtime_log_path),
        },
    )

    render_script_path = CURRENT_DIR / "render_memory_demo.py"
    replay_logger.log(
        "render_demo_start",
        {
            "render_script_path": str(render_script_path),
            "input_results_path": str(output_path),
        },
    )
    render_proc = subprocess.run(
        [sys.executable, str(render_script_path), "--input", str(output_path)],
        cwd=str(CURRENT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if render_proc.stdout.strip():
        print(render_proc.stdout.rstrip())
    if render_proc.stderr.strip():
        print(render_proc.stderr.rstrip(), file=sys.stderr)

    replay_logger.log(
        "render_demo_finished",
        {
            "return_code": render_proc.returncode,
            "stdout_preview": truncate_text(render_proc.stdout, 300),
            "stderr_preview": truncate_text(render_proc.stderr, 300),
        },
    )
    if render_proc.returncode != 0:
        raise RuntimeError(f"render_memory_demo.py 执行失败，returncode={render_proc.returncode}")

    generated_html_path = output_path.with_name(output_path.stem + "_demo.html").resolve()
    print(f"可点击绝对路径: {generated_html_path}")
    print(f"可点击浏览器链接: {to_clickable_uri(generated_html_path)}")

    try:
        os.startfile(str(generated_html_path))
        print(f"已自动打开浏览器页面: {generated_html_path}")
    except Exception as exc:
        print(f"自动打开浏览器失败，请手动打开: {generated_html_path} -> {exc}")


if __name__ == "__main__":
    main()
