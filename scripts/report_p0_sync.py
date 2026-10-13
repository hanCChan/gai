#!/usr/bin/env python3
"""Export grouped evidence and P0-C decision without signal/model access."""
import argparse
import csv
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def dump(path, x):Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csvout(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def hardware(rx):return 'RTL-SDR' if rx.startswith('rtl') else {'b200_mini_1':'USRP B200 mini','b210_1':'USRP B210','pluto_1':'ADALM-Pluto'}[rx]


def group(rows, **identity):
    r=dict(**identity,packets=len(rows),reliable_phase_count=sum(x['reliable_symbol_phase']=='True' for x in rows),
           quiet_prefix_candidates=sum(x['left_quiet_candidate']=='True' for x in rows),
           packet_start_identified_count=sum(x['packet_start_identified']=='True' for x in rows),
           up_chirp_blocks=sum(int(x['up_chirp_blocks']) for x in rows),down_chirp_blocks=sum(int(x['down_chirp_blocks']) for x in rows))
    for k in ['nominal_coherence','chirp_r2','chirp_phase_modulo_samples','candidate_radius_samples',
              'inter_chirp_phase_q95_samples','bandwidth_cycles_per_sample','prefix_energy_ratio','suffix_energy_ratio','rms']:
        a=np.array([float(x[k]) for x in rows])
        for suffix,q in [('min',0),('q05',.05),('median',.5),('q95',.95),('max',1)]:r[k+'_'+suffix]=float(np.quantile(a,q))
    return r


def run(source,out):
    if out.exists():raise ValueError('Use a new report directory')
    reg=json.loads((ROOT/'configs/p0_sync/registration.json').read_text())
    rows=list(csv.DictReader((source/'packets.csv').open()))
    verification=json.loads((source/'independent_verification.json').read_text());assert verification['passed']
    syn=json.loads((source/'synthetic_validation.json').read_text());assert syn['passed']
    assert len(rows)==2000 and {r['receiver'] for r in rows}==set(reg['source_receivers'])
    rxstats=[group([x for x in rows if x['receiver']==rx],receiver=rx,hardware_type=hardware(rx),fold=next(i for i,z in enumerate(reg['source_receiver_folds']) if rx in z)) for rx in reg['source_receivers']]
    cells=[group([x for x in rows if x['receiver']==rx and int(x['tx_raw_label'])==tx],receiver=rx,hardware_type=hardware(rx),tx_raw_label=tx) for rx in reg['source_receivers'] for tx in range(31,41)]
    types=[group([x for x in rows if hardware(x['receiver'])==h],hardware_type=h,receiver_count=sum(hardware(rx)==h for rx in reg['source_receivers'])) for h in dict.fromkeys(hardware(rx) for rx in reg['source_receivers'])]
    folds=[group([x for x in rows if int(x['fold'])==i],fold=i,receiver_count=5) for i in range(2)]
    global_stats=group(rows,scope='ten_source_train_Rx_only')
    # This report deliberately requires evidence beyond chirp shape/phase.
    # The experiment has no independent time reference and found no broad
    # candidate spread; neither is a classification-effect experiment.
    observed_stable=(global_stats['reliable_phase_count']==2000 and global_stats['candidate_radius_samples_max']<=8
                     and global_stats['inter_chirp_phase_q95_samples_max']<=8 and global_stats['down_chirp_blocks']==0)
    if not observed_stable:raise ValueError('Evidence differs; assess observability under frozen rules before assigning a decision')
    decision={'decision':'STOP_NO_OBSERVABLE_SYNC','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'A0':'completed_with_explicit_source_gaps','A1':'completed; conditional chirp phase identifiable; natural synchronization uncertainty unestablished',
              'B':'not_entered_due_to_observability_gate','C':'completed_as_stop_decision_and_problem_definition',
              'natural_sync_uncertainty_observable':False,'meaning_of_false':'Not reliably demonstrated under present data/reference/diagnostic scope; not proof of absence',
              'effect_on_existing_model':'not_tested','standard_methods_solve_it':'not_tested',
              'new_training_runs':0,'gpu_runs':0,'model_loads':0,'new_unique_source_packets':2000,
              'development_signal_reads':0,'locked_signal_reads':0,'day3_day4_signal_reads':0,'osu_signal_reads':0,'large_dataset_downloads':0,
              'baseline_explains_decision_supported':False,'mechanism_supported_decision_supported':False,
              'reasons':['No independent packet-start or alignment-error reference in available clipped HDF5; upstream preprocessing unknown',
                         'All 2000 packets have stable conditional chirp shapes; modulo phase 2–4 samples is not true STO',
                         'Candidate radii 3–4 samples and inter-chirp phase q95 at most 2.65; no broad uncertainty observed in this scope',
                         'No reliable downchirp structure under frozen shape diagnostic; one low-prefix-energy example alone is not an onset reference'],
              'next_stage_authorized':False,'stopped_after_P0':True}
    out.mkdir(parents=True)
    for name in ['packets.csv','chirp_blocks.csv','synthetic_trials.csv','synthetic_validation.json','source_manifest.json','signal_access_log.json','independent_verification.json','representative_profiles.json','representative_candidate_curves.json','status.json','registration.json','seal.json']:
        shutil.copy2(source/name,out/name)
    csvout(out/'per_RX.csv',rxstats);csvout(out/'per_RX_TX.csv',cells);csvout(out/'per_hardware_type.csv',types);csvout(out/'per_source_fold.csv',folds)
    dump(out/'summary.json',global_stats);dump(out/'decision.json',decision)
    status=json.loads((out/'status.json').read_text());status.update(status=decision['decision'],A0=decision['A0'],B=decision['B'],C=decision['C'],stopped_after_P0=True)
    dump(out/'status.json',status)
    provenance={'rejection_source_sha256':hashlib.sha256(Path('/home/keyan/private/p0_sync_20261013/rejection.txt').read_bytes()).hexdigest(),
                'rejection_original_verified':True,'rejected_manuscript_pdf_local_status':'not_found; existing OOB PDF has a different title',
                'full_additional_brief_local_status':'not_received; user pasted execution decision is operative',
                'private_documents_in_this_report_or_bundle':False,'original_worktree_head':subprocess.check_output(['git','-C','/home/keyan/code-y-p2a','rev-parse','HEAD'],text=True).strip(),
                'original_worktree_status':subprocess.check_output(['git','-C','/home/keyan/code-y-p2a','status','--porcelain'],text=True).strip(),
                'same_remote_tree_at_start':True}
    dump(out/'provenance.json',provenance)
    for name in ['manifest.json','extra_manifest.json','shen60_structure_manifest.json']:
        shutil.copy2(ROOT/'outputs/p0_sync/20261013_a0_literature'/name,out/('literature_'+name))
    figure(out,reg,cells,rows)
    lines=['# 第二篇论文 P0：同步机制审计决策报告','',
           '**决策：STOP_NO_OBSERVABLE_SYNC。完成本轮 P0 后停止，不进入 P1 或新模型训练。**','',
           '在限定的2,000个源训练包中，重复上 chirp 结构清晰、符号相位候选稳定；但没有可靠的自然同步误差真值或独立起点参照。当前证据不足以支撑“同步不确定性导致设备身份损失”的论文机制。不能据此声称同步永远无影响。','',
           '## 三个必须回答的问题','',
           '|问题|本轮结论|','|---|---|',
           '|同步不确定性是否真实可观测？|本轮未可靠建立。可观测的是裁剪窗内的重复结构和条件性符号相位，不是完整包起点或真实同步误差。|',
           '|是否影响已有模型？|尚未检验。A1关口未通过，未加载权重或读入旧预测。|',
           '|现有标准方法是否已解决？|尚未检验。本轮不能选择 STOP_BASELINE_EXPLAINS，也不能声称有标准方法无法解决的剩余缺口。|','',
           '## 执行范围与冻结顺序','',
           '基于 code-y-p2a 的 eab01d2，在独立 worktree `code-y-p0-sync`、分支 `experiment/p0-sync-mechanism` 执行。原目录未提交修改为零，HEAD保持不变。远端提交历史不同，但起始文件树相同；无合并/推送。','',
           '检测规则、精确2,000行、合成标准及代码先封存，随后验证合成信号，再读取十个白名单训练文件。每 RX×TX 从 G1 历史200包索引中抽取20包，两个五RX源折各1,000包，无信号驱动的参数拟合。原始全文件 SHA256 与 G1 一致；只对选定2,000行解码并分析 IQ。','',
           '四开发RX、六锁定RX、任何日期子集、OSU原始IQ均未读取。没有真实IQ循环移位、额外CFO补偿、重采样、数据集下载、CNN推理或训练。独立复核重读了同一2,000包集合中的100个分组代表及1个低首部能量例；没有增加新包。','',
           '## 检测器验证','',
           '360个解析长上chirp裁剪正对照（20/30/40dB，三种静态CFO）通过登记标准：九个单元误差q95均为1样本，可靠覆盖率均100%。白噪声、单音、下chirp各100个负对照，误接收均0/100。','',
           '这仅验证整数符号相位在AWGN下的可恢复性，不验证完整帧起点、多径、SFO或接收机群时延。8样本标准在名义配置下对应一个chip；条件性物理参数未当成本地已确认设置。','',
           '## 真实源数据观测','',
           f'- 2,000/2,000包满足冻结的条件性形状/相位可靠规则；1024点相关相干度中位数 {global_stats["nominal_coherence_median"]:.6f}，最小 {global_stats["nominal_coherence_min"]:.6f}。',
           f'- 锯齿拟合R²中位数 {global_stats["chirp_r2_median"]:.6f}；相位模1024的估计为2–4样本（2:1308包，3:689包，4:3包）。',
           '- R²候选半径为3–4样本；逐chirp相位差q95的跨包最大值2.65样本。它们是模板诊断指标，不是置信区间或真实STO。',
           '- 在冻结的块形状规则下，16,000个名义块均为up方向，可靠down块为0。没有看到可验证的SFD/完整帧结构。',
           '- 十个HDF5及data/label/CFO/SNR属性为空。局部Fs/BW/SF、上游同步、CFO/重采样、实际日期/session和外部时间锚点仍未确认。','',
           '|RX|硬件类型|包数|相干度中位数|相位范围（样本）|逐chirp差异q95最大值|',
           '|---|---|---:|---:|---:|---:|']
    for r in rxstats:
        lines.append(f'|{r["receiver"]}|{r["hardware_type"]}|{r["packets"]}|{r["nominal_coherence_median"]:.6f}|{r["chirp_phase_modulo_samples_min"]:.0f}–{r["chirp_phase_modulo_samples_max"]:.0f}|{r["inter_chirp_phase_q95_samples_max"]:.2f}|')
    lines += ['', '十RX中七台是RTL，不能当十种独立硬件架构。两个源留出折结论一致；逐RX/TX、硬件类型和折统计均已单列。包/块/两折不作为新增独立采集，不报告伪重复的显著性。','',
              '首部低能量候选只有一个：b210_1/TX36/row4358，前128点能量比0.006075。独立复核确认没有前导精确零段，能量沿前256点渐变；这可以是波形幅度/AGC/衰落现象，不能单独确定包起点，也未在多个源RX复现。因此未删除该包或把它当新同步失效。','',
              '## 停止原因与证据边界','',
              '作者文献记录同步、提取前导和CFO处理，源样本又具有非常集中的chirp相位；这与数据已进行上游裁剪相容，但不足以独立确认其精度或具体流程。几样本的模板相位包含离散化、平滑、滤波与硬件差异，不能当作待消除的纯误差。','',
              '仅有重复up chirp时无法确定哪一个符号是整个包的开端；本轮也没有连续窗外IQ、时间戳或可信同步真值。现有源证据不足以区分自然同步估计不确定性与硬件/信道波形差异，故A1未满足进入B的条件。没有用人工jitter或扩大样本制造研究缺口。','',
              '## P0-C：问题定义、强基线与下一步价值','',
              '若未来重新开启该方向，具体问题应是：在具备真实边界参照和可校准自然同步不确定性的条件下，固定RFFI模型的身份决策是否随误差变化，标准同步/CFO/CIS/PIR后是否仍有剩余影响。须控制RX、TX与信号质量，先验证检测误差再讨论身份保持。当前尚没有可登记为新算法贡献的证据。','',
              '优先补齐已有数据的采集/裁剪/补偿代码、真实帧边界与同步真值资料，再考虑小规模边界完整样本。仅增加60TX裁剪前导或扩大到20,000包，不会自动补上窗外信息，因此当前不值得在这条路线上启动GPU PoC。A开放集仍为后续任务选择，C聚合反转未扩展为并行实验。','',
              '标准同步、两阶段CFO及CIS已有先行工作；[文献差异矩阵](../../../docs/p0_sync/LITERATURE_MATRIX.md)逐项标明已取得全文、仅摘要与未复现部分。PIR仅核验出版社摘要，不能声称已排除其全文机制或完成性能对照。','',
              '完整拒稿信已核验；被拒RF-HSTU原稿PDF本地未找到，现有OOB PDF是另一研究。资料缺口和60TX结构/成本估算见[溯源记录](../../../docs/p0_sync/DATA_AND_REVIEW_PROVENANCE.md)。未发表材料保持仓库外私有。','',
              '## 复核、证据与复现','',
              f'独立直接相关程序覆盖100个RX×TX单元，相位差0样本，R²最大差 {verification["r2_absolute_error_max"]:.3g}，相干度最大差 {verification["coherence_absolute_error_max"]:.3g}；2000行与历史索引子集、封存SHA和访问范围一致。该复核验证数值与作用范围，不代替机制证据。','',
              '[证据索引](EVIDENCE_INDEX.md)、[全部包统计](packets.csv)、[RX×TX统计](per_RX_TX.csv)、[决策JSON](decision.json)、[诊断图](figures/source_observability.png)。','',
              '复现使用独立Python环境、CPU和新的空输出目录：','', '```bash',
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/keyan/venvs/shen-stage1/bin/python scripts/audit_p0_sync.py --output outputs/p0_sync/reproduction_new",
              'OPENBLAS_NUM_THREADS=1 /home/keyan/venvs/shen-stage1/bin/python scripts/verify_p0_sync.py outputs/p0_sync/reproduction_new',
              '```','',
              '运行路径及原始输入依赖见 registration.json；不复制旧输出覆盖本报告。主审计CPU墙钟约72.7秒（含10文件来源hash），训练0次。公开材料检索和报告制作另计。','',
              '**本轮已经停止；未注册或自动开启新的GPU实验。**','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    index=['# P0 证据索引','', '|文件|用途|','|---|---|',
           '|REPORT.md / decision.json|P0-C结论及三问题答复|','|registration.json / seal.json|结果前规则与代码SHA|',
           '|packets.csv / chirp_blocks.csv|2000包、16000块逐项指标|','|per_RX.csv / per_RX_TX.csv|10 RX和100个RX×TX单元|',
           '|per_hardware_type.csv / per_source_fold.csv|硬件类型及固定两折，非独立采集统计|','|synthetic_trials.csv / synthetic_validation.json|合成时移与负对照|',
           '|source_manifest.json / signal_access_log.json|十源文件来源、元数据和只读访问|',
           '|independent_verification.json|100单元独立重算、1低能量案例及复核访问|',
           '|representative_profiles.json / representative_candidate_curves.json|每RX×TX首个选定行的频率轮廓/候选曲线，无原始IQ|',
           '|literature_*manifest.json|公开资料来源/检索SHA与失败记录|','|provenance.json|私有拒稿材料hash、缺失原稿及原目录状态|',
           '|figures/*.png / *.pdf|独立图件|','|ARTIFACT_MANIFEST.json|文件完整性SHA|','',
           '私有拒稿信/原稿、原始IQ和CNN权重不在报告或ZIP。完整准确样本行号见仓库 configs/p0_sync/sample_manifest.json。',
           '文献全文/代码快照只存本地outputs/p0_sync/20261013_a0_literature；索引记录获取成功/失败，ZIP不包含他人全文。','']
    (out/'EVIDENCE_INDEX.md').write_text('\n'.join(index))
    artifact={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'ARTIFACT_MANIFEST.json',artifact)
    # Archive includes exact selected row manifest and frozen source; no raw IQ.
    z=out/'P0_sync_evidence.zip'
    with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=z:f.write(p,'evidence/'+str(p.relative_to(out)))
        seal=json.loads((ROOT/'configs/p0_sync/seal.json').read_text())
        source_files=set(seal['files'])|{'configs/p0_sync/seal.json','scripts/verify_p0_sync.py','scripts/report_p0_sync.py','src/p0_sync/__init__.py','docs/p0_sync/LITERATURE_MATRIX.md','docs/p0_sync/DATA_AND_REVIEW_PROVENANCE.md','docs/p0_sync/GIT_BASELINE.json'}
        for rel in sorted(source_files):
            f.write(ROOT/rel,'source/'+rel)
    dump(out/'BUNDLE_SHA256.json',{'file':z.name,'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest()})
    print(json.dumps(decision,ensure_ascii=False))


def figure(out,reg,cells,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir()
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    for ax,key,title,vmax in [(axes[0,0],'chirp_phase_modulo_samples_median','Median modulo chirp phase (samples)',8),
                              (axes[0,1],'inter_chirp_phase_q95_samples_max','Maximum inter-chirp phase q95 (samples)',8)]:
        matrix=np.array([[next(c[key] for c in cells if c['receiver']==rx and c['tx_raw_label']==tx) for tx in range(31,41)] for rx in reg['source_receivers']])
        im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap='viridis',aspect='auto');ax.set_xticks(range(10),range(31,41));ax.set_yticks(range(10),reg['source_receivers'],fontsize=8);ax.set_xlabel('Stored TX label');ax.set_title(title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.8)
    axes[1,0].hist([float(r['nominal_coherence']) for r in rows],bins=35,color='#245c82');axes[1,0].set_title('1024-sample lag coherence');axes[1,0].set_xlabel('Normalized magnitude');axes[1,0].set_ylabel('Packets')
    counts=[sum(float(r['chirp_phase_modulo_samples'])==x for r in rows) for x in (2,3,4)]
    axes[1,1].bar([2,3,4],counts,color='#418e78');axes[1,1].set_xticks([2,3,4]);axes[1,1].set_title('Conditional symbol-phase estimates');axes[1,1].set_xlabel('Samples modulo 1024; NOT true STO');axes[1,1].set_ylabel('Packets')
    for x,n in zip([2,3,4],counts):axes[1,1].text(x,n+15,str(n),ha='center')
    fig.suptitle('P0-A1: 2,000 source-only packets; no packet-start ground truth',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(folder/('source_observability.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.output.resolve())
