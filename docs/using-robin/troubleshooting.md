# Troubleshooting

!!! abstract "What this page covers"
    Common **browser** issues: can’t load the page, sign-in, **Unknown sample**, watched folders errors, dark mode, **Workflow** menu, failed reports. For server install issues, see the [README — Common issues](https://github.com/LooseLab/ROBIN/blob/main/README.md#common-issues) and [CLI reference](../cli/index.md).

---

## I can’t open the page {#i-cant-open-the-page}

| Check | Action |
|-------|--------|
| **URL** | Copy exactly from the terminal or your admin (`http://…`, correct **port**, often **8081**). |
| **Network** | Same LAN/VPN if ROBIN runs on another machine. |
| **Firewall** | IT may need to allow the port. |
| **Process running** | The web UI is only up while **`robin workflow`** is running. Restart or ask your admin. |
| **Work directory** | Some setups need **`--work-dir`** for the main dashboard — [What happens at startup](../getting-started/startup.md). |

---

## I can’t sign in {#i-cant-sign-in}

- Use the **same password** as for the web UI (often set at first **GUI password** prompt in the terminal).  
- **Caps Lock** off; retry after a short wait.  
- If no password was ever set, run ROBIN **from a terminal** once — [At startup](../getting-started/startup.md).  
- Ask an admin to reset: **`robin password set`** or **`robin users set-password <username>`**.  

---

## It says “Unknown sample” {#unknown-sample}

ROBIN only knows samples it has **seen** (or that are already tracked). If the ID is wrong or BAMs have not arrived:

- Wait for **BAMs** in the watched folder.  
- Open **All samples** and **View** the correct **row**.  
- Confirm the **library ID** matches MinKNOW.  

---

## “Watched folders” mentions Ray or the workflow {#watched-folders-ray}

That screen **adds or removes input folders**. It only works in configurations that support it. Most clinical users do **not** change this mid-run — ask bioinformatics if you see an error.

---

## Dark mode looks wrong {#dark-mode}

Toggle **Dark Mode** in the menu off and on. If a **chart** is wrong, **refresh** or leave the sample and reopen — plots sometimes update after a moment.

---

## The “Workflow” menu item doesn’t work {#workflow-menu}

Use **Activity Monitor** — that is the **workflow monitor** in the standard setup.

---

## Report or download failed {#report-failed}

- Wait for spinners / notifications to finish.  
- Retry; large PDFs can take time.  
- Ask your admin to check **disk space** and **write permissions** on the output folder.  

---

## “Failed to load ITD hotspots” in the log {#itd-hotspots-missing}

Every ITD job logs `Failed to load ITD hotspots for panel …: No such file or directory: …/resources/itd_hotspots.hg38.json`. The curated hotspot file is missing from the install; ITD calling is skipped until it is restored (all other analyses are unaffected).

Regenerate it from the packaged GENCODE annotation:

```bash
python scripts/make_itd_hotspots.py
```

This writes `src/robin/resources/itd_hotspots.hg38.json` with one window per curated gene (FLT3, BCOR, KIT, NPM1, CALR, CEBPA, EGFR, ERBB2, JAK2, NOTCH1); genes absent from the active `--target-panel` are ignored at run time. Restart the workflow afterwards. The file is required in every `[itd] region_mode` (`hotspots`, `panel` and `both` all read it), so regenerating it is the fix in all cases.

---

## Still stuck {#still-stuck}

- [README — Common issues](https://github.com/LooseLab/ROBIN/blob/main/README.md#common-issues)  
- [Command-line reference](../cli/index.md) (staff who run ROBIN)  
