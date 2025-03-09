import numpy as np
import torch
import torch.nn as nn
import video
import Frank_Wolfe as fw
from time import time
import audio
import matplotlib.pyplot as plt
import random

torch.set_default_dtype(torch.float64)

def test_attack(method, dataset, y_tar, eps, max_iter, gap_tol , model):
    successful_attacks = 0
    tot_number_of_iterations = 0
    i = 0
    t0 = time()
    lowest_fs = []

    for x_att, y_att in dataset:
        print("Attack: ", i)
        x_adv, fs = FW_attack(method, x_att, y_tar, eps, max_iter, gap_tol,model)
        #attack_summary(x_adv,x_att,y_tar,False,model)
        lowest_fs.append(fs[-1])
        tot_number_of_iterations += len(fs)
        with torch.no_grad():
            log_probs = model(x_adv)
            y_adv = log_probs.argmax().item()

        if y_adv == y_tar:
            successful_attacks += 1
        i = i + 1

    t1 = time()

    total_attacks = len(dataset)
    success_rate = successful_attacks / total_attacks
    avg_num_iterations = tot_number_of_iterations / total_attacks
    elapsed_time = t1-t0
    avg_attack_time = elapsed_time/total_attacks
    print('num tests: ', total_attacks)
    print('success rate: ', success_rate)
    print("elapsed time: ", elapsed_time)
    print("tot num of iterations: ", tot_number_of_iterations)
    #print("lowest value: ", fs[-1])

    return success_rate, avg_attack_time, avg_num_iterations, lowest_fs

def attack_summary(x_adv, x_att, y_tar, show, model):

    with torch.no_grad():
        p_y_x_att = torch.exp(model(x_att)).flatten().numpy()
        p_y_x_adv = torch.exp(model(x_adv)).flatten().numpy()

    y_att = np.argmax(p_y_x_att).item()
    y_adv = np.argmax(p_y_x_adv).item()

    # Print attack summary
    print("Attack Summary:")
    print("attack status: ", "success" if y_tar == y_adv else 'failure')
    print("y_tar: ", y_tar)
    print("y_att: ", y_att)
    print("y_adv: ", y_adv)
    print(f"p(y_tar|x_att): {p_y_x_att[y_tar]:.3f}")
    print(f"p(y_adv|x_att): {p_y_x_att[y_adv]:.3f}")
    print(f"p(y_att|x_att): {p_y_x_att[y_att]:.3f}")
    print(f"p(y_tar|x_adv): {p_y_x_adv[y_tar]:.3f}")
    print(f"p(y_adv|x_adv): {p_y_x_adv[y_adv]:.3f}")
    print(f"p(y_att|x_adv): {p_y_x_adv[y_att]:.3f}")

    if show:
        if x_att.shape == (1, 1, 28, 28):
            x_att_img = x_att.squeeze().detach().numpy()
            x_adv_img = x_adv.squeeze().detach().numpy()
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            def plot_with_histogram(ax, img, probabilities, title, color):
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                ax.set_title(title)
                inset_ax = ax.inset_axes([-0.05, 0, 1.1, 0.4])
                inset_ax.set_facecolor('none')
                for spine in inset_ax.spines.values():
                    spine.set_visible(False)
                bars = inset_ax.bar(range(10), probabilities, color=color, alpha=0.4)
                for bar in bars:
                    bar.set_edgecolor(color)
                inset_ax.set_ylim(0, 1)
                inset_ax.set_xticks(range(10))
                #inset_ax.set_yticks([])
                #inset_ax.set_xlabel('Class', fontsize=8)
                #inset_ax.set_ylabel('Prob', fontsize=8)

            plot_with_histogram(axes[0], x_att_img, p_y_x_att, "Attack Example", 'red')
            plot_with_histogram(axes[1], x_adv_img, p_y_x_adv, "Adversarial Example", 'red')

            plt.tight_layout()
            plt.show()

        else:
            audio.play_sound(x_att)
            audio.play_sound(x_adv)

class TargetedObjective(nn.Module):
    def __init__(self, network: nn.Module, target_label: int):
        super(TargetedObjective, self).__init__()
        self.network = network
        self.target_label = target_label

    def forward(self, x: torch.Tensor):
        x = x.reshape(self.network.input_shape)
        log_probs = self.network(x)
        return -log_probs.flatten()[self.target_label]

def FW_attack(method, x_att, y_tar, eps, max_iter, gap_tol, model):
    x_0 = x_att.reshape(-1)
    f = TargetedObjective(model,y_tar)

    C1 = fw.LInf_Ball(x_0, eps)
    C2 = fw.Box(model.domain_bounds)
    C = C1 * C2

    x_star, fs = fw.UFW(method, f, x_0, C, max_iter, gap_tol, 1e-5)

    if x_att.shape == (1, 1, 28, 28):
        x_adv = torch.clip(x_star, 0,1)
        x_adv = x_adv.reshape(1, 1, 28, 28)
    else:
        x_adv = torch.clip(x_star, -1,1)
        x_adv = x_adv.reshape(1, 1, -1)

    return x_adv, fs

def do_plots1(attack_set, y_tar, epss, max_iter, gap_tol, model):
    res_fw = []
    res_pwfw = []
    props = []

    for eps in epss:
        res_fw.append(test_attack("fw", attack_set[0:num_tests], y_tar, eps, max_iter, gap_tol, model))
        res_pwfw.append(test_attack("pwfw", attack_set[0:num_tests], y_tar, eps, max_iter, gap_tol, model))

        den = sum(1 for x, y in zip(res_fw[-1][3], res_pwfw[-1][3]) if x != y)
        if den == 0: den = -1
        prop = sum(1 for x, y in zip(res_fw[-1][3], res_pwfw[-1][3]) if x != y and x > y) / den
        props.append(prop)

    y_labels = ["avg accuracy", "avg time (seconds)", "avg iterations", "proportion FW > PWFW"]
    titles = [
        "Accuracy vs Epsilon for FW and PWFW",
        "Average Time vs Epsilon for FW and PWFW",
        "Average Iterations vs Epsilon for FW and PWFW",
        "Proportion FW >= PWFW vs Epsilon"
    ]

    avg_acc_fw = sum(result[0] for result in res_fw) / len(res_fw)
    avg_acc_pwfw = sum(result[0] for result in res_pwfw) / len(res_pwfw)
    avg_time_fw = sum(result[1] for result in res_fw) / len(res_fw)
    avg_time_pwfw = sum(result[1] for result in res_pwfw) / len(res_pwfw)
    avg_iter_fw = sum(result[2] for result in res_fw) / len(res_fw)
    avg_iter_pwfw = sum(result[2] for result in res_pwfw) / len(res_pwfw)
    avg_prop = sum(props) / len(props)

    print("Accuracy plot:")
    for eps, fw_acc, pwfw_acc in zip(epss, [result[0] for result in res_fw], [result[0] for result in res_pwfw]):
        print(f"eps {eps}: FW acc: {fw_acc:.2f}, PWFW acc: {pwfw_acc:.2f}")
    print("avg accuracy FW: ", avg_acc_fw, "avg accuracy PWFW: ", avg_acc_pwfw)

    print("\nElapsed time plot:")
    for eps, fw_time, pwfw_time in zip(epss, [result[1] for result in res_fw], [result[1] for result in res_pwfw]):
        print(f"eps {eps}: FW time: {fw_time:.2f}, PWFW time: {pwfw_time:.2f}")
    print("avg time FW: ", avg_time_fw, "avg time PWFW: ", avg_time_pwfw)

    print("\nAverage iterations plot:")
    for eps, fw_iter, pwfw_iter in zip(epss, [result[2] for result in res_fw], [result[2] for result in res_pwfw]):
        print(f"eps {eps}: FW iter: {fw_iter:.2f}, PWFW iter: {pwfw_iter:.2f}")
    print("avg of avg iterations FW: ", avg_iter_fw, "avg of avg iterations PWFW: ", avg_iter_pwfw)

    print("\nProportion FW >= PWFW plot:")
    for eps, prop in zip(epss, props):
        print(f"eps {eps}: Proportion FW >= PWFW: {prop:.2f}")
    print("avg proportion FW >= PWFW: ", avg_prop)

    for el in range(3):
        plt.figure(figsize=(8, 6))
        res_fw_el = [result[el] for result in res_fw]
        res_pwfw_el = [result[el] for result in res_pwfw]

        plt.plot(epss, res_fw_el, marker='o', label='FW')
        plt.plot(epss, res_pwfw_el, marker='s', label='PWFW')

        if el == 0:  # Accuracy plot
            plt.axhline(y=avg_acc_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Accuracy')
            plt.axhline(y=avg_acc_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Accuracy')

        elif el == 1:  # Time plot
            plt.axhline(y=avg_time_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Time')
            plt.axhline(y=avg_time_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Time')

        elif el == 2:  # Average Iterations plot
            plt.axhline(y=avg_iter_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Iterations')
            plt.axhline(y=avg_iter_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Iterations')

        plt.xticks(ticks=epss, labels=[f"{eps:.5f}" for eps in epss], fontsize=8)

        plt.xlabel('epsilon')
        plt.ylabel(y_labels[el])
        plt.title(titles[el])
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epss, props, marker='d', color='purple', label='Proportion FW >= PWFW')
    plt.axhline(y=avg_prop, color='red', linestyle='--', linewidth=1, label='Avg Proportion FW >= PWFW')

    plt.xticks(ticks=epss, labels=[f"{eps:.5f}" for eps in epss],fontsize=8)

    plt.xlabel('epsilon')
    plt.ylabel(y_labels[3])
    plt.title(titles[3])
    plt.legend()
    plt.grid(True)
    plt.show()

def do_plots2(attack_set, y_tar, eps, max_iter, gap_tol, model):
    fw_res = []
    pwfw_res = []
    props = []
    digits = list(range(10))
    filtered_digits = []

    for digit in digits:
        if digit == y_tar:
            continue

        filtered_digits.append(digit)

        digit_filtered_dataset = [(x, y) for x, y in attack_set if y == digit]

        fw_res.append(test_attack("fw", digit_filtered_dataset[0:num_tests], y_tar, eps, max_iter,gap_tol, model))
        pwfw_res.append(test_attack("pwfw", digit_filtered_dataset[0:num_tests], y_tar, eps, max_iter, gap_tol, model))

        den = sum(1 for x, y in zip(fw_res[-1][3], pwfw_res[-1][3]) if x != y)
        if den == 0: den = -1
        prop = sum(1 for x, y in zip(fw_res[-1][3], pwfw_res[-1][3]) if x != y and x > y) / den
        props.append(prop)

    y_labels = ["avg accuracy", "avg time (seconds)", "avg iterations", "proportion FW > PWFW"]
    titles = [
        "Accuracy vs Digit for FW and PWFW",
        "Time vs Digit for FW and PWFW",
        "Average Iterations vs Digit for FW and PWFW",
        "Proportion FW > PWFW vs Digit"
    ]

    avg_acc_fw = sum(result[0] for result in fw_res) / len(fw_res)
    avg_acc_pwfw = sum(result[0] for result in pwfw_res) / len(pwfw_res)
    avg_time_fw = sum(result[1] for result in fw_res) / len(fw_res)
    avg_time_pwfw = sum(result[1] for result in pwfw_res) / len(pwfw_res)
    avg_iter_fw = sum(result[2] for result in fw_res) / len(fw_res)
    avg_iter_pwfw = sum(result[2] for result in pwfw_res) / len(pwfw_res)
    avg_prop = sum(props) / len(props)

    print("Accuracy plot:")
    for digit, fw_acc, pwfw_acc in zip(filtered_digits, [result[0] for result in fw_res], [result[0] for result in pwfw_res]):
        print(f"Digit {digit}: FW acc: {fw_acc:.2f}, PWFW acc: {pwfw_acc:.2f}")
    print("avg accuracy FW: ", avg_acc_fw, "avg accuracy PWFW: ", avg_acc_pwfw)

    print("\nElapsed time plot:")
    for digit, fw_time, pwfw_time in zip(filtered_digits, [result[1] for result in fw_res], [result[1] for result in pwfw_res]):
        print(f"Digit {digit}: FW time: {fw_time:.2f}, PWFW time: {pwfw_time:.2f}")
    print("avg time FW: ", avg_time_fw, "avg time PWFW: ", avg_time_pwfw)

    print("\nAverage iterations plot:")
    for digit, fw_iter, pwfw_iter in zip(filtered_digits, [result[2] for result in fw_res], [result[2] for result in pwfw_res]):
        print(f"Digit {digit}: FW iter: {fw_iter:.2f}, PWFW iter: {pwfw_iter:.2f}")
    print("avg of avg iterations FW: ", avg_iter_fw, "avg of avg iterations PWFW: ", avg_iter_pwfw)

    print("\nProportion FW > PWFW plot:")
    for digit, prop in zip(filtered_digits, props):
        print(f"Digit {digit}: Proportion FW > PWFW: {prop:.2f}")
    print("avg proportion FW > PWFW: ", avg_prop)

    for el in range(3):
        plt.figure(figsize=(8, 6))
        fw_values = [result[el] for result in fw_res]
        pwfw_values = [result[el] for result in pwfw_res]

        plt.plot(filtered_digits, fw_values, marker='o', label='FW')
        plt.plot(filtered_digits, pwfw_values, marker='s', label='PWFW')

        if el == 0:
            plt.axhline(y=avg_acc_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Accuracy')
            plt.axhline(y=avg_acc_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Accuracy')

        elif el == 1:
            plt.axhline(y=avg_time_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Time')
            plt.axhline(y=avg_time_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Time')

        elif el == 2:
            plt.axhline(y=avg_iter_fw, color='blue', linestyle='--', linewidth=1, label='Avg FW Iterations')
            plt.axhline(y=avg_iter_pwfw, color='orange', linestyle='--', linewidth=1, label='Avg PWFW Iterations')

        plt.xlabel('digit')
        plt.ylabel(y_labels[el])
        plt.title(titles[el])
        plt.legend()
        plt.grid(True)
        plt.xticks(filtered_digits)
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(filtered_digits, props, marker='d', color='purple', label='Proportion FW > PWFW')
    plt.axhline(y=avg_prop, color='red', linestyle='--', linewidth=1, label='Avg Proportion FW > PWFW')
    plt.xlabel('digit')
    plt.ylabel(y_labels[3])
    plt.title(titles[3])
    plt.legend()
    plt.grid(True)
    plt.xticks(filtered_digits)
    plt.show()

def do_plots3(attack_set, y_tar, epss, max_iter, gap_tol, model):
    num_digits = 10
    num_eps = len(epss)

    fig, axes = plt.subplots(num_digits - 1, num_eps, figsize=(10, 10))

    for i in range(num_digits):
        if i == y_tar:
            continue

        for e, eps in enumerate(epss):
            digit_filtered_dataset = [(x, y) for x, y in attack_set if y == i]
            x_att, y_att = random.choice(digit_filtered_dataset)
            x_adv, _ = FW_attack("fw", x_att, y_tar, eps, max_iter, gap_tol, model)

            with torch.no_grad():
                log_probs = model(x_adv)
                y_adv = log_probs.argmax().item()
                success = y_adv == y_tar

            ax = axes[i if i < y_tar else i - 1, e]
            x_adv_img = x_adv.squeeze().detach().numpy()
            ax.imshow(x_adv_img, cmap='gray', vmin=0, vmax=1)

            border_color = 'green' if success else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == num_digits - 1 or (i == num_digits - 2 and y_tar == num_digits - 1):
                ax.set_xlabel(f"{eps:.2f}", fontsize=12)

            if e == 0:
                ax.set_ylabel(f"{i}", fontsize=12, rotation=0, ha='right', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    module = video #audio
    model = module.model_init(load_model=True)
    attack_set = module.attack_set_init(model=model)

    y_tar = 7
    y_tar_filtered_attack_set = [(x, y) for x, y in attack_set if y != y_tar]
    eps = 0.15 #0.001
    x_att = y_tar_filtered_attack_set[3][0]
    max_iter = 500
    gap_tol = 0.00005
    num_tests = 50

    x_adv, ffw = FW_attack("fw", x_att, y_tar, eps, max_iter, gap_tol, model)
    attack_summary(x_adv, x_att, y_tar, True, model)