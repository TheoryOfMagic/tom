import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.optimize import curve_fit

cmap = cm.get_cmap("viridis")


def count_graphs(n):
    edges = n * (n - 1) / 2
    return 2**edges


# for i in [3,4,5,6,7,8,9,10]:
#    print(f"an {i} sided shape has {count_graphs(i)} possible symbols")


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


# draw_spell(level = 3,rang = 150,dtype = 'fire',area = 'Sphere',title = "Fireball")
# draw_spell(level = 0,rang = 60,dtype = 'radiant',
#           area = 'None',title = "Sacred Flame")


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
    for i in np.arange(int((math.ceil(len(x) / 2)))):
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


def binomial_coeff(n, k):
    return math.factorial(n) / (math.factorial(n - k))


def distance_from_origin(x, y):
    d = np.sqrt(x**2 + y**2)
    return d


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


def genbin(n, bs=""):
    if n - 1:
        genbin(n - 1, bs + "0")
        genbin(n - 1, bs + "1")
    else:
        print("1" + bs)


def generate_binary_strings(bit_count):
    binary_strings = []

    def genbin(n, bs=""):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return binary_strings


def cycle_list(l, loops=1):
    n = len(l)
    for t in range(loops):
        l = [l[(i + 1) % n] for i in range(n)]
    return l


def decode_shape(
    in_array, n, k=1, radius=1, start_angle=None, color="k", label=None, base="Polygon"
):
    if start_angle == None:
        start_angle = np.pi / n

    if base == "Polygon":
        small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

        x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    elif base == "Line":
        x = np.arange(0, n + 1)
        y = np.zeros((1, n))

    elif base == "Quadratic":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [x_**2 for x_ in x]
    elif base == "Quadratic2":
        x = [0]
        while len(x) < n:
            if -x[-1] in x:
                x.append(-x[-1] + 1)
            else:
                x.append(-x[-1])
        y = [x_**2 for x_ in x]
    elif base == "SemiCircular":
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "QuarterCircular":
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "CubicFunction":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [0.1 * x_**3 + -0.75 * x_ for x_ in x]

    labelled = False
    for i in range(n):
        if in_array[i] == 1:
            if labelled == False:
                plt.plot(
                    [x[i], x[(i + k) % n]],
                    [y[i], y[(i + k) % n]],
                    "-",
                    color=color,
                    label=label,
                )
                labelled = True
            else:
                plt.plot(
                    [x[i], x[(i + k) % n]], [y[i], y[(i + k) % n]], "-", color=color
                )
        elif in_array[i] == 2:
            offset = 0.01
            plt.plot(
                [x[i], x[(i + k) % n]],
                [y[i], y[(i + k) % n]],
                "-",
                color=color,
                label=label,
            )
            plt.plot(
                [x[i] + offset, x[(i + k) % n] + offset],
                [y[i] + offset, y[(i + k) % n] + offset],
                "-",
                color=color,
            )
        else:
            plt.plot(
                [x[i], x[(i + k) % n]],
                [y[i], y[(i + k) % n]],
                ":",
                linewidth=0.5,
                color=color,
            )
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.axis("scaled")
    plt.axis("off")
    # plt.show()


def generate_unique_combinations(L):
    combinations = generate_binary_strings(L)
    non_repeating = [combinations[0]]
    for i in range(len(combinations)):
        if i % round(0.1 * len(combinations)) == 0:
            print(f"{100*i/len(combinations):.1f}%")

        ref = list(combinations[i])
        N = len(ref)
        test = 0
        for j in range(len(non_repeating)):
            for n in range(N):
                if cycle_list(list(non_repeating[j]), loops=n + 1) == ref:
                    test += 1
                # else:
                #    print(combinations[j], ref)

        if test == 0:
            non_repeating.append(combinations[i])

    # print(non_repeating)
    for i in np.arange(len(non_repeating)):
        non_repeating[i] = [int(s) for s in list(non_repeating[i])]
    return non_repeating


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
    data_trans = data.T
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


def draw_centre_circle(P, Q, thetas=None):
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]
    a = (x1 + x2) / 2
    b = (y1 + y2) / 2
    r = np.sqrt((a - x1) ** 2 + (b - y1) ** 2)

    if thetas == "Full":
        theta = np.linspace(0, 2 * np.pi, 150)
    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)
        if y2 < y1:
            theta0, theta1 = theta1 + np.pi, theta0 + np.pi

        theta = np.linspace(theta0, theta1, 150)
    X2 = r * np.cos(theta) + a
    Y2 = r * np.sin(theta) + b

    return (X2, Y2, a, b)


def draw_non_centre_circle(P, Q, b, radius=0, thetas=None):
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]

    b2 = -b

    delta = x1**2 - x2**2 + y1**2 - y2**2
    a = (delta - 2 * (y1 - y2) * b) / (2 * (x1 - x2))
    a2 = (delta - 2 * (y1 - y2) * b2) / (2 * (x1 - x2))

    r = np.sqrt((x1 - a) ** 2 + (y1 - b) ** 2)
    r2 = np.sqrt((x1 - a2) ** 2 + (y1 - b2) ** 2)
    if r2 <= r:
        a = a2
        b = b2
        r = r2

    if thetas == "Full":
        theta1 = np.linspace(0, 2 * np.pi, 150)

    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)

        theta02 = theta0
        theta12 = theta1

        while theta1 < theta0:
            theta0 -= 2 * np.pi

        while theta02 < theta12:
            theta12 -= 2 * np.pi

        arc1 = r * (theta1 - theta0)
        arc2 = r * (theta02 - theta12)

        # print(f'Arc1: {arc1}\n Arc2: {arc2} \n {(arc1+arc2)/(2*np.pi*r)}')
        if arc1 < arc2 or np.sqrt(b**2) < 1:
            theta = np.linspace(theta1, theta0, 150)
        else:
            theta = np.linspace(theta02, theta12, 150)

    X = r * np.cos(theta) + a
    Y = r * np.sin(theta) + b
    return (X, Y, a, b)


def decode_shape_circular(
    in_array,
    k=1,
    radius=1,
    start_angle=None,
    label=None,
    color="k",
    centered=True,
    s=0,
    base="Polygon",
):
    if np.sqrt(s**2) < 1 and "Quadratic" in base:
        raise Exception("Exception: Quadratic shapes requires s >= 1")
    n = len(in_array)
    # print(in_array)
    if start_angle == None:
        start_angle = np.pi / n

    if base == "Polygon":
        small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

        x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    elif base == "Line":
        x = np.arange(0, n)
        y = np.zeros((1, n))[0]
    elif base == "Quadratic":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [x_**2 for x_ in x]
    elif base == "Quadratic2":
        x = [0]
        while len(x) < n:
            if -x[-1] in x:
                x.append(-x[-1] + 1)
            else:
                x.append(-x[-1])
        y = [x_**2 for x_ in x]

    elif base == "SemiCircular":
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "QuarterCircular":
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "CubicFunction":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [0.1 * x_**3 + -0.75 * x_ for x_ in x]
        # print(x,y)
    # small_angle = [start_angle + i * 2*np.pi/n for i in np.arange(1,n+1)]

    # x,y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    if label == None:
        labelled = True
    else:
        labelled = False

    for i in range(n):
        P = [x[i], y[i]]
        Q = [x[(i + k) % n], y[(i + k) % n]]
        if centered == True:
            X, Y, a, b = draw_centre_circle(P, Q)
        else:
            X, Y, a, b = draw_non_centre_circle(P, Q, radius=radius, b=s)
        # plt.plot(a,b,"x",color = color)
        if in_array[i] == 1:
            if labelled == False:
                plt.plot(X, Y, label=label, color=color)
                labelled = True
            else:
                plt.plot(X, Y, color=color)

        else:
            # P = [x[i],y[i]]
            # Q = [x[(i+k)%n],y[(i+k)%n]]
            # X,Y,a,b = draw_centre_circle(P,Q)
            plt.plot(X, Y, ":", color=color, linewidth=0.3)
            # plt.plot(a,b,".")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")

    plt.axis("scaled")
    plt.axis("off")


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
    data_trans = data.T
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


"""P = [1.2246467991473532e-16, -1.0]
Q = [-0.5406408174555972, -0.8412535328311814]
b = 1
X,Y,a,b = draw_non_centre_circle(P,Q,b,thetas = None)#

plt.plot(X,Y)
plt.plot(a,b,".")
plt.plot(P[0],P[1],"x",c = 'r')
plt.plot(Q[0],Q[1],"x",c = 'b')
plt.show()

P = [0.5406408174555978, -0.8412535328311811]
Q = [1.2246467991473532e-16, -1.0]
b = 1
X,Y,a,b = draw_non_centre_circle(P,Q,b,thetas = None)#

plt.plot(X,Y)
plt.plot(a,b,".")
plt.plot(P[0],P[1],"x",c = 'r')
plt.plot(Q[0],Q[1],"x",c = 'b')
plt.show()"""
# n = 5
# in_array = [1,0,1,1,0]
# for k in range(4):
#    plt.subplot(1,4,k+1)
#    plt.title(f'{str(in_array)}, k = {k+1}')
#    decode_shape(in_array,n,k=k+1)
# plt.show()
# n = 11

# print(x2,y2)
# decode_shape_circular([1,0,1,1,0,1,0,1,1,0,1],k = 3,centered = False, s = 1, base = "Polygon")
# plt.show()
"""N = 11
if os.path.isfile(f'Uniques/{N}.npy'):
    non_repeating = np.load(f'Uniques/{N}.npy')
else:
    non_repeating = generate_unique_combinations(N)
    non_repeating = np.array(non_repeating)
    np.save(f"Uniques/{N}.npy",non_repeating)

S = [0.7,1,5,10,100]

for j in range(len(S)):
    s = S[j]
    plt.subplot(1,len(S),j+1)
    for i in range(len(S)):
        input_array = random.choice(non_repeating[:10])
        decode_shape_circular(input_array,k=i+1,centered = False,s = s)
    #plt.vlines(0,-2,2)
    #plt.hlines(0,-2,2)
    plt.title(f'b = {s}')

plt.savefig("varying b.png",transparent = True)
plt.show()"""

# keys = ['level', 'school', 'damage','area_types','range']
# for i in range(len(keys)):
#    save_feature_key(color = 'k',key = keys[i],k = i+1,shape = "Straight",s = 1,base = "Polygon")

# i = 0

# draw_feature_key(color = 'k',key = keys[i],k = i+1,shape = "Straight",s = 1,base = "Polygon")
# save_feature_key(color = 'k',key = keys[i],k = i+1,shape = "Straight",s = 1,base = "Polygon")
# draw_spell_search("Fireball",shape = "Non_Centred", breakdown = True,s = 1,base = "CubicFunction",save = True)
# draw_spell_search("Random",shape = "Non_Centred", breakdown = True,s = 1,base = "SemiCircular")


"""N = 8
bins = generate_binary_strings(N)
uniq = generate_unique_combinations(N)
groups = [[]]*len(uniq)
#print(groups)
for i,u in enumerate(uniq):
    #print(u)
    #print(groups[i])
    groups[i] = [u]
#print(uniq)

for b in bins:
    b = [int(b_) for b_ in list(b)]
    for n in np.arange(1,N):
        #print(n)
        #print(cycle_list(b,n))
        if cycle_list(b,n) in uniq:
            #print("here")
            index = uniq.index(cycle_list(b,n))
            if b not in groups[index]:
                groups[index].append(b)"""

# for g in groups:
# print(np.array(g))
"""diff = 0
for g in groups:
    if len(g) != N and len(g) != 1:
        diff += len(g)
        #print(g)
        #print(len(g))
        #print("-------------")
    else:
        pass
        #print(len(g))
        #print("-------------")
print((2**N - 2 + diff)/N + 2, len(uniq))
n = 2
print(1/N * (n**6 + n**3 + 2*n**2 + 2*n))
f = lambda n:sum(1-any(i>int(b[j::-1]+b[:j:-1],2)or j*(i>=int(b[j:]+b[:j],2))for j in range(n))for i in range(2**n)for b in[bin(i+2**n)[3:]])
print(f(N))
#print((2**N - 2)/N + 2)
#print((2**N - 2)%N)
#print((2**N - 2 + (2**N - 2)%N)/N + 2)
#print(len(uniq))

Expect = []
Actual = []
Correction = []
N_list = []
for N in np.arange(3,14):
    N_list.append(N)
    #print(f'\subsection{{n = {N}}}')
    #print("\begin{multicols}{4}")
    if os.path.isfile(f'Uniques/{N}.npy'):
        non_repeating = np.load(f'Uniques/{N}.npy')
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy",non_repeating)
    #for nr in non_repeating:
        #print(nr,"\n")
    #print("\end{multicols}")
    print(f'----------{N}------------')
    #print((2**N-2)%N)
    print(len(non_repeating))
    Expect.append((2**N - 2)/N + 2)
    Actual.append(len(non_repeating))
    Correction.append(len(non_repeating) - ((2**N - 2)/N + 2))
    #print((2**N - 2)/N + 2)

x,y = [],[]
for i in np.arange(len(Correction)):
    if Correction[i] != 0:
        x.append(N_list[i])
        y.append(Correction[i])
#print(Correction)
plt.plot(x,y,".")
plt.show()"""

"""n = 0
N = 4
folders = os.listdir(path = "keys")
print(folders)
doc_folders = ["Areas","Damage_Type","Level","Range","School"]
for fold in folders:
    n = 0
    doc_fold = doc_folders[folders.index(fold)]
    print(f"\n\subsection{{ {doc_fold} }}\n")
    files = os.listdir("keys/" + fold)
    print(
        r"\begin{figure}[H]","\n","\t\centering\n",)
    for file in files:
            n += 1
            print(f"\t\includegraphics[scale = 0.2]{{Dict_Files/{doc_fold}/{file}}}\n",
            "\t\label{{fig:{file}}}\hfill",)
            if  n%N ==0:
                print(f'\t',r'\\[\smallskipamount]')
                n = 0
            
                
    print("\end{figure}\n")"""

in_arrays = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]
for i in range(len(in_arrays)):
    print(i)
    decode_shape_circular(
        in_arrays[i],
        k=i + 1,
        radius=1,
        start_angle=None,
        label=None,
        color="k",
        centered=True,
        s=0,
        base="Polygon",
    )
plt.title("Find Familiar")
plt.show()

for i in range(len(in_arrays)):
    print(i)
    decode_shape(
        in_arrays[i],
        n=11,
        k=i + 1,
        radius=1,
        start_angle=None,
        label=None,
        color="k",
        base="Polygon",
    )
plt.title("Find Familiar")
plt.show()
