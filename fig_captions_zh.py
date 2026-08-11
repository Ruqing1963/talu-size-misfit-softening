"""Chinese figure captions, keyed identically to fix_tex.FIGS."""
FIGS_ZH = {
 "FIG1": ("fig1_elastic_fields.pdf", 1.0,
  r"由弹性偶极张量给出的溶质--位错相互作用。(a) $r=10$\,\AA\ 处的角向依赖；"
  r"各向同性 Stroh 解复现闭式解 $A\sin\theta/r$，误差 $2.6\times10^{-8}$。"
  r"(b) 刃型位错的各向异性场。(c) 各向异性与各向同性之比：振幅仅变化 "
  r"$-1.8\%$，但在滑移面附近角向形状偏差达 $\pm30\%$，而那正是 "
  r"$\mathrm{d}U/\mathrm{d}x$ 控制钉扎的区域。"),
 "FIG7": ("fig7_atmosphere_2d.pdf", 1.0,
  r"600\,K、$c_0=2$\,at.\% 下由守恒 Kawasaki 蒙特卡洛得到的二维柯氏气团。"
  r"(a) 弹性场与平均场 $c=1/2$ 等值线，后者是与滑移面相切、直径为 $R_c$ 的圆。"
  r"(b) 平衡占据。(c) 径向分布：由于溶质间吸引，蒙特卡洛结果在过渡壳层高于"
  r"平均场 Fermi--Dirac 分布。(d) Warren--Cowley 参数呈\textbf{环状}，"
  r"在饱和核心处消失。"),
 "FIG8": ("fig8_warren_cowley.pdf", 0.60,
  r"Warren--Cowley $\alpha_1$ 随局部成分的变化。序参量在 $c\to0$ 与 $c\to1$ "
  r"两端均消失：饱和气团不携带化学有序，因为其中已无化学。蒙特卡洛点高于"
  r"均匀准化学曲线，是因为跨越浓度梯度的取样窗口记录到了虚假的有序。"),
 "FIG2": ("fig2_edge_depinning.pdf", 1.0,
  r"刃型位错脱钉。(a) $N_z=128$ 下临界应力随时效程度的变化；$\mu/30$ 以上的"
  r"灰色区域在力学上不可达，故完全时效的气团不允许脱钉。(b) 溶质间相互作用"
  r"使 $\tau_c$ 相对 Bernoulli 近似最高提高 $1.8$ 倍。"),
 "FIG5": ("fig5_screw_kmc.pdf", 1.0,
  r"(a) 气团并不钉扎速率控制型螺位错：任意时效程度下 "
  r"$\tau_c^{\rm screw}/\tau_c^{\rm edge}<10^{-3}$。(b) 扭折对动力学蒙特卡洛："
  r"随机 Lu 场在 100 至 450\,K 的所有温度下都提高流变应力。"),
 "FIG6": ("fig6_kappa_threshold.pdf", 1.0,
  r"(a) 唯象化学软化项 $\kappa$ 仅在超过 150\,K 下的 $\kappa^*=0.17$\,eV 时"
  r"才使符号反转；250\,K 时任何取值都不足够，因为纯 Ta 已进入非热区。"
  r"(b) 退火平均涨落项 $\sigma^2/2k_BT$ 与淬火极值极限、以及与 $\kappa^*$ 的对比。"),
 "FIG3": ("fig3_convergence.pdf", 1.0,
  r"收敛性。(a) 网格、应力斜坡与截断半径三项参数收敛至优于 1\%。"
  r"(b) 线长度不收敛，也不应当收敛：最弱环节统计给出对数衰减，至 $N_z=2048$ "
  r"仍无平台。(c) 构型间散布 9.6\%，据此可报告的精度为两位有效数字。"),
 "FIG4": ("fig4_sensitivity.pdf", 1.0,
  r"对两个仍待第一性原理输入的量的敏感度：$\tau_c\propto U_{\rm cap}^{0.91}$ "
  r"与 $\propto\Omega_{\rm rel}^{1.26}$，对比 Varvenne--Curtin 的解析指数 "
  r"$4/3$。早期草稿曾报告 $U_{\rm cap}$ 的指数为 $0.20$，那是 5.1 节所述"
  r"不连续核心截断造成的伪像。"),
 "FIGS1": ("figS1_dd_map.pdf", 0.62,
  r"用于第一性原理计算的螺型偶极的微分位移图，验证了两个反号核心已生成、"
  r"且滑移区位于二者之间。周期性残差为 $|\mathbf{b}|$ 的 0.03\%。"),
}
