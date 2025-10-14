import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from utils.color_palettes import ALL_PALETTES
import matplotlib as mpl
import platform

# --- 크로스 플랫폼 한글 글꼴 설정 ---
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
mpl.rcParams['axes.unicode_minus'] = False


def create_elevation_heatmap(elevation_array, bounds, stats, palette_key="terrain"):
    """
    고도 데이터의 히트맵 시각화를 생성합니다.

    매개변수:
    -----------
    elevation_array : numpy.ndarray
        고도 데이터 배열.
    bounds : tuple
        분석 영역의 경계 (왼쪽, 아래, 오른쪽, 위).
    stats : dict
        고도 데이터의 통계.
    palette_key : str, optional
        사용할 색상 팔레트의 키 (기본값: 'spectral').

    반환값:
    --------
    matplotlib.figure.Figure
        시각화를 포함하는 그림 객체.
    """
    # 적절한 색상 팔레트 가져오기
    if palette_key not in ALL_PALETTES:
        palette_key = "spectral"  # 기본값

    colors = ALL_PALETTES[palette_key]["colors"]
    cmap = LinearSegmentedColormap.from_list(f"{palette_key}_cmap", colors, N=256)

    # 더 큰 크기로 그림 생성
    fig, ax = plt.subplots(figsize=(10, 8))

    # 고도 데이터를 히트맵으로 플롯
    im = ax.imshow(
        elevation_array,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=bounds,
        vmin=stats["min"],
        vmax=stats["max"],
    )

    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("고도 (m)", rotation=270, labelpad=20)

    # 더 나은 고도 시각화를 위해 등고선 추가
    levels = np.linspace(stats["min"], stats["max"], 15)
    contour = ax.contour(
        elevation_array, levels=levels, colors="k", alpha=0.3, extent=bounds
    )

    # 라벨 및 제목 설정
    ax.set_xlabel("동쪽 좌표 (m)")
    ax.set_ylabel("북쪽 좌표 (m)")
    ax.set_title("고도 분석", fontweight="bold", fontsize=14)

    # 통계 주석 추가
    stats_text = (
        f"최소: {stats['min']:.2f} m\n"
        f"최대: {stats['max']:.2f} m\n"
        f"평균: {stats['mean']:.2f} m\n"
        f"면적: {stats.get('area', 'N/A')} sq km"
    )

    # 통계를 위한 상자 생성
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
        fontsize=10,
    )

    # 참조용 그리드 라인 추가
    ax.grid(True, linestyle="--", alpha=0.3)

    # 고도 범위 범례 추가
    handles = [
        mpatches.Patch(color=colors[0], label=f'최소 ({stats["min"]:.1f} m)'),
        mpatches.Patch(
            color=colors[5],
            label=f'중간 ({stats["min"] + (stats["max"]-stats["min"])/2:.1f} m)',
        ),
        mpatches.Patch(color=colors[-1], label=f'최대 ({stats["max"]:.1f} m)')
    ]
    ax.legend(handles=handles, loc="lower right", title="고도 범위")

    # 레이아웃 조정
    plt.tight_layout()

    return fig