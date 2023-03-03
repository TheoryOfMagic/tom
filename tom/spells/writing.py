from __future__ import annotations

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from matplotlib import cm

from tom.util.math import count_graphs
from tom.util.math import generate_unique_combinations

cmap = cm.get_cmap("viridis")


logger = structlog.get_logger()


def draw_shape(n, radius=1, start_angle=None):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]
    print(len(small_angle))
    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    for i in np.arange(len(x)):
        for j in np.arange(len(x))[i + 1 :]:
            plt.plot([x[i], x[j]], [y[i], y[j]], c="k")
    plt.title(f"{n} sides: {count_graphs(n)} possibilities")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.axis("scaled")
    plt.show()


def draw_spell(level, rang, dtype, area, n=12, start_angle=None, radius=1, title=None):
    ranges = [0, 5, 10, 30, 60, 90, 100, 150, 300, 500, 1000, 5280]
    levels = list(np.arange(0, 12))
    area_types = [
        "None",
        "ST",
        "MT",
        "Wall",
        "Sphere",
        "Cyl",
        "Cone",
        "Line",
        "Cube",
        "Square",
        "Circle",
    ]
    dtypes = [
        "bludgeoning",
        "cold",
        "poison",
        "thunder",
        "acid",
        "necrotic",
        "force",
        "radiant",
        "psychic",
        "lightning",
        "fire",
        "slashing",
    ]

    i_range = ranges.index(rang)
    i_levels = levels.index(level)
    i_area = area_types.index(area)
    i_dtype = dtypes.index(dtype)

    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    atributes = [i_dtype, i_levels, i_range, i_area]
    labels = [
        f"damage type: {dtype}",
        f"level: {level}",
        f"range: {rang}ft",
        f"area_type: {area}",
    ]
    for k in np.arange(len(atributes)):
        # print(k)
        i = atributes[k]
        j = (atributes[k] + k + 1) % 12
        # print(i,j)
        plt.plot([x[i], x[j]], [y[i], y[j]], label=labels[k])
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.title(title)
    plt.axis("scaled")
    plt.axis("off")
    plt.legend()
    plt.show()


def draw_explain(k, array, n=12, radius=1, start_angle=None, title=None, show=True):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    for it in np.arange(n):
        i = it
        j = (it + k) % n
        try:
            plt.plot([x[i], x[j]], [y[i], y[j]], label=array[it], c="k")
        except:
            plt.plot([x[i], x[j]], [y[i], y[j]], label="Unassigned", c="k")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.title(title)
    plt.axis("scaled")
    if show == True:
        plt.legend()
        plt.show()


def draw_edge_groups(n, radius=1, start_angle=None):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    for i in np.arange(int(math.ceil(len(x) / 2))):
        c = cmap(i / (math.ceil(len(x) / 2)))
        print(i)
        # for j in np.arange(len(x))[i+1:]:
        #   plt.plot([x[i],x[j]],[y[i],y[j]],c = 'k')
        for j in np.arange(len(x)):
            if j == 0:
                plt.plot(
                    [x[j], x[(j + i) % n]],
                    [y[j], y[(j + i) % n]],
                    "-.",
                    color="k",
                    label=i,
                )
            else:
                plt.plot([x[j], x[(j + i) % n]], [y[j], y[(j + i) % n]], "--", color=c)

    plt.legend()
    plt.show()


def calculate_all_possible_shapes(n, k):
    # n = #of sides
    # k = #of sides activated
    existing_sets = []
    all_possible = []

    max_join = n - k
    zeros = np.zeros(k)
    zeros[0] = max_join
    all_possible.append(list(zeros))
    existing_sets.append(set(all_possible[-1]))
    if k == 1:
        return all_possible
    elif k == n:
        return [0]
    while zeros[0] != zeros[-1]:
        print(zeros)
        for i in np.arange(len(zeros) - 1):
            print(i)
            if zeros[i + 1] < zeros[i]:
                zeros[i] -= 1
                zeros[i + 1] += 1

                if set(zeros) not in existing_sets:
                    all_possible.append(list(zeros))

                    existing_sets.append(set(all_possible[-1]))
                    print(zeros)
    return all_possible


def draw_spell_search(
    search_name,
    output_name="output.csv",
    start_angle=None,
    radius=1,
    color=False,
    breakdown=False,
    shape="Straight",
    s=0,
    base="Polygon",
    save=False,
):
    data = pd.read_csv(output_name)
    data_trans = data.T
    if search_name == "Random":
        search_name = random.choice(data["name"])

    indices = [i for i in np.arange(len(data)) if data_trans[i]["name"] == search_name]
    final = data_trans[indices[0]]
    for i in indices:
        d = data_trans[i]

        if d["cantrip_scaling"] <= final["cantrip_scaling"]:
            final = d

    ranges = sorted(set(data["range"]))

    areas = set(data["area_types"].astype(str))
    areas = sorted(list(set(data["area_types"])), key=lambda x: str(x))

    schools = sorted(set(data["school"]))
    levels = np.arange(10)
    dtypes = sorted(set(data["damage"]))

    lens = [len(areas), len(levels), len(ranges), len(dtypes), len(schools)]
    n = max(lens) + 2

    if start_angle == None:
        start_angle = np.pi / n

    # small_angle = [start_angle + i * 2*np.pi/n for i in np.arange(1,n+1)]

    # x,y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    print(final)
    atributes = []
    atribute_labels = []

    atributes.append(list(levels).index(final["level"]))
    atribute_labels.append(f"level: {final['level']}")

    atributes.append(list(schools).index(final["school"]))
    atribute_labels.append(f"School: {final['school']}")

    atributes.append(list(dtypes).index(final["damage"]))
    atribute_labels.append(f"Damage Type: {final['damage']}")

    atributes.append(list(areas).index(final["area_types"]))
    atribute_labels.append(f" Area Type: {final['area_types']}")

    atributes.append(list(ranges).index(final["range"]))
    atribute_labels.append(f" Range: {final['range']}")

    N = 2 * len(atributes) + 1
    # print("Attributes: " ,atributes)
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    if breakdown == True:
        for i in np.arange(int(N / 2)):
            ax1 = int(N / 2)
            plt.subplot(1, ax1, i + 1)
            plt.title(atribute_labels[i])
            j = atributes[i] + 1

            if shape == "Straight":
                decode_shape(
                    non_repeating[j],
                    n=N,
                    k=i + 1,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[j],
                    k=i + 1,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[j],
                    k=i + 1,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape Undefined")
        plt.show()
    for i in np.arange(math.floor(N / 2)):
        plt.title(search_name)
        j = atributes[i] + 1
        # print("nr:",non_repeating[j])
        if shape == "Straight":
            decode_shape(
                non_repeating[j],
                n=N,
                k=i + 1,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                base=base,
            )
        elif shape == "Circle":
            decode_shape_circular(
                non_repeating[j],
                k=i + 1,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                base=base,
            )
        elif shape == "Non_Centred":
            decode_shape_circular(
                non_repeating[j],
                k=i + 1,
                s=s,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                centered=False,
                base=base,
            )
        else:
            print("Shape Undefined")
        # decode_shape(non_repeating[j+1],n = N, k = i+1,color = cmap(i/math.floor(N/2)),label = atribute_labels[i])
        # plt.legend(loc='upper left')
    if save == True:
        plt.savefig(f"Diagrams/{search_name}_{shape}_{base}.png", transparent=True)
    plt.show()


def draw_feature_key(
    N=11,
    output_name="output.csv",
    key="level",
    k=1,
    s=1,
    start_angle=None,
    radius=1,
    color=None,
    shape="Straight",
    base="Polygon",
):
    data = pd.read_csv(output_name)
    print(data.keys())
    if key == "area_types":
        features = set(data["area_types"].astype(str))
        features = sorted(list(set(data["area_types"])), key=lambda x: str(x))
    else:
        features = sorted(set(data[key]))
    print(len(features))
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    for i in np.arange(len(features)):
        plt.subplot(math.ceil(len(features) / 4), 4, i + 1)
        # print(non_repeating[i])
        if color == None:
            plt.title(features[i])
            if shape == "Straight":
                decode_shape(
                    non_repeating[i + 1],
                    n=N,
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centre":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape not supported")
        else:
            if shape == "Straight":
                decode_shape(non_repeating[i + 1], n=N, k=k, color="k", base=base)
            elif shape == "Circle":
                decode_shape_circular(non_repeating[i + 1], k=k, color="k", base=base)
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[i + 1], k=k, s=s, color="k", centered=False, base=base
                )
            else:
                print("Shape not supported")
            plt.title(features[i])
    plt.show()


def save_feature_key(
    N=11,
    output_name="output.csv",
    key="level",
    k=1,
    s=1,
    start_angle=None,
    radius=1,
    color=None,
    shape="Straight",
    base="Polygon",
):
    data = pd.read_csv(output_name)
    print(data.keys())
    if key == "area_types":
        features = set(data["area_types"].astype(str))
        features = sorted(list(set(data["area_types"])), key=lambda x: str(x))
        features = [
            str(f)
            .replace("C", "Cube")
            .replace("N", "Cone")
            .replace("L", "Line")
            .replace("R", "Circle")
            .replace("S", "Sphere")
            .replace("MT", "Multiple-Targets")
            .replace("W", "Wall")
            .replace("Y", "Cylinder")
            .replace("SphereT", "Single-Target")
            .replace("nan", "None")
            .replace("Q", "Square")
            for f in features
        ]
        # print(features)
        # input()
    else:
        features = sorted(set(data[key]))
    print(len(features))
    if not os.path.exists(f"keys/{key}_{shape}_{base}"):
        os.mkdir(f"keys/{key}_{shape}_{base}")
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    for i in np.arange(len(features)):
        # plt.subplot(math.ceil(len(features)/4),4,i+1)
        # print(non_repeating[i])
        if color == None:
            plt.title(features[i])
            if shape == "Straight":
                decode_shape(
                    non_repeating[i + 1],
                    n=N,
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centre":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape not supported")
        else:
            if shape == "Straight":
                decode_shape(non_repeating[i + 1], n=N, k=k, color="k", base=base)
            elif shape == "Circle":
                decode_shape_circular(non_repeating[i + 1], k=k, color="k", base=base)
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[i + 1], k=k, s=s, color="k", centered=False, base=base
                )
            else:
                print("Shape not supported")
            plt.title(f"{features[i]}   {non_repeating[i+1]}")
        print(str(features[i]).replace("/", "-"))
        plt.savefig(
            f'keys/{key}_{shape}_{base}/{str(features[i]).replace("/","-")}.png',
            transparent=True,
        )
        plt.close()
