import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def add_node(ax, xy, r=0.12, text="", fontsize=10, lw=1.5):
    x, y = xy
    c = Circle((x, y), r, fill=False, lw=lw)
    ax.add_patch(c)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return c

def add_edge(ax, p1, p2, text=None, text_offset=(0, 0.06),
             rad=0.0, lw=1.3, arrow=True, fontsize=10):
    """
    p1,p2: (x,y)
    rad: 弧度，0 为直线；正负控制弧线朝向（像你图里外圈回环就用较大 rad）
    """
    style = "Simple,tail_width=0.6,head_width=6,head_length=8" if arrow else "-"
    conn = f"arc3,rad={rad}"
    patch = FancyArrowPatch(
        p1, p2,
        arrowstyle=style if arrow else "-",
        connectionstyle=conn,
        lw=lw,
        color="black",
        mutation_scale=1.0
    )
    ax.add_patch(patch)

    if text is not None:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx + text_offset[0], my + text_offset[1], str(text),
                ha="center", va="center", fontsize=fontsize)
    return patch

def demo_like_your_figure():
    fig, ax = plt.subplots(figsize=(7,6))

    # ---- 节点坐标（你可以自己改）----
    inputs = [(0.2, y) for y in [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9]]
    hL = (-0.6, 0.15)
    hR = ( 0.6, 0.15)
    out = (0.0, 1.25)

    # ---- 画节点 ----
    for i, p in enumerate(inputs):
        add_node(ax, p, r=0.09, text="")
    add_node(ax, hL, r=0.14, text="-11", fontsize=11)
    add_node(ax, hR, r=0.14, text="-11", fontsize=11)
    add_node(ax, out, r=0.14, text="64", fontsize=11)

    # ---- 输入到 hidden 的平行连线 + 权重标签（示例）----
    w_left  = [14.2, -3.6, 7.2, -7.2, 3.6, -14.2]
    w_right = [-14.2, 3.6, -7.1, 7.1, -3.6, 14.2]

    for p, w in zip(inputs, w_left):
        add_edge(ax, p, hL, text=w, rad=0.0, text_offset=(-0.05, 0.03), lw=1.1)
    for p, w in zip(inputs, w_right):
        add_edge(ax, p, hR, text=w, rad=0.0, text_offset=(0.05, 0.03), lw=1.1)

    # ---- hidden 到 output 的“外圈回环”两条弧线（像你图里 -8.8 那样）----
    add_edge(ax, hL, out, text="-8.8", rad=0.55, text_offset=(-0.15, 0.08), lw=1.4)
    add_edge(ax, hR, out, text="-8.8", rad=-0.55, text_offset=(0.15, 0.08), lw=1.4)

    # ---- 标题文字（像图里的 unit label）----
    ax.text(0.0, 1.05, "output unit", ha="center", va="bottom", fontsize=11)
    ax.text(-0.95, 0.15, "hidden\nunit", ha="center", va="center", fontsize=11)
    ax.text(0.95, 0.15, "hidden\nunit", ha="center", va="center", fontsize=11)
    ax.text(0.0, -0.9, "input units", ha="center", va="top", fontsize=11)

    # 画布设置：论文风关键
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.45)
    ax.axis("off")

    plt.tight_layout()
    plt.show()

# 运行示例
demo_like_your_figure()
