[← Back to Home](/)

# Various Tips and How To Dos...

**Table of Contents**:
- [How to Reduce PDF File Size](#how-to-reduce-pdf-file-size)
- [How to Resize Video Recording](#how-to-resize-video-recording)
- [How to Bootstrap a New GitHub Project](#how-to-bootstrap-a-new-github-project)
- [How to use GitHub Templates](#how-to-use-github-templates)

## How to Reduce PDF File Size
To reduce the size of a PDF from the command line—especially one made of heavy scanned images—the industry standard tool you want is **Ghostscript** (`gs` - Ghostscript is a tool for creating, viewing, and transforming PostScript and PDF documents).


If you don't have it installed, you can easily grab it via Homebrew:
```bash
brew install ghostscript
```

Once installed, here is the command to compress your scanned PDF:
```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile=compressed_output.pdf original_scanned.pdf
```

### What makes this work:
The magic happens with the **`-dPDFSETTINGS`** flag, which dictates the compression level. 
* **`/ebook`**: (150 dpi) This is usually the sweet spot for scanned documents, offering a much smaller file size while keeping the text perfectly readable.
* **`/screen`**: (72 dpi) Use this if you need the absolute smallest file possible and don't mind a drop in visual quality.
* **`/printer`**: (300 dpi) Use this if you need to retain high quality for physical printing, though it won't compress as aggressively.

The `/ebook` tag itself is a fixed preset. It acts as a shortcut that tells Ghostscript to apply a whole bundle of internal settings, which happen to include downsampling images to exactly 150 DPI. You can't pass a custom DPI directly into the `/ebook` tag (like `/ebook:200` or something similar).

However, if you want complete control to dial in a very specific DPI—say you want 200 DPI because 150 is a little too blurry, or 100 DPI because you need it even smaller—**you absolutely can**. 

To do this, you just drop the `-dPDFSETTINGS` preset and manually tell Ghostscript what resolution to use for the images inside the PDF. 

Here is how you would run the command to force a custom DPI (using 200 as an example):

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dNOPAUSE -dQUIET -dBATCH \
  -dDownsampleColorImages=true -dColorImageResolution=200 \
  -dDownsampleGrayImages=true -dGrayImageResolution=200 \
  -dDownsampleMonoImages=true -dMonoImageResolution=200 \
  -sOutputFile=custom_dpi_output.pdf original_scanned.pdf
```

### What changed in this command:
Instead of relying on a one-size-fits-all preset, we are explicitly turning on downsampling for all image types (Color, Grayscale, and Monochrome) and manually setting the resolution for each to `200`. You can change that `200` to whatever number fits your exact needs.

---

## How to Resize Video Recording

**FFmpeg** is the industry standard and will do exactly what you want in seconds.

**Install FFmpeg:**

  * **Mac:** `brew install ffmpeg`
  * **Windows:** `winget install ffmpeg`

**Command A: Letterbox (Fit inside box with black bars)**

```bash
ffmpeg -i input.mp4 -vf "scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:(ow-iw)/2:(oh-ih)/2" -c:a copy output_padded.mp4
```

**Command B: Center Crop (Fill box, cut sides)**

```bash
ffmpeg -i input.mp4 -vf "crop=1020:1020,scale=1024:1024" -c:a copy output_cropped.mp4
```

*(Note: We crop to 1020 first because that is your video's native height, then scale slightly to 1024).*

Here is the exact command to scale the width to **1024** while keeping the aspect ratio automatically:

```bash
ffmpeg -i input.mp4 -vf scale=1024:-2 -c:a copy output_1024.mp4
```

### Breakdown of the command:

  * **`-vf scale=1024:-2`**: This is the magic part.
      * `1024`: Sets your fixed width.
      * `-2`: Tells FFmpeg to automatically calculate the height to maintain the aspect ratio **AND** ensure the height is an even number (divisible by 2). This is critical because many codecs (like H.264) will fail if you try to use an odd number for the height (e.g., 765 pixels).
  * **`-c:a copy`**: Copies the audio without re-encoding it. This makes the process much faster and prevents any audio quality loss.

**Result:**
Your new video will likely be **1024x766**.

---

## How to Bootstrap a New GitHub Project

Your logic is spot on - it’s essentially "forking" the project manually to give yourself a clean slate while keeping the history if you want it (or clearing it if you don't).

Here is the step-by-step workflow to get your new repository linked to your existing codebase.

### 1. Clone the Existing Project
First, clone the original repository into a **new folder** on your Mac.

```bash
git clone https://github.com/username/old-repo.git my-new-project
cd my-new-project
```

### 2. Create the New Repo on GitHub
* Go to [GitHub](https://github.com) and create a **New Repository**.
* Give it your new name (e.g., `my-new-project`).
* **Important:** Do **not** initialize it with a README, license, or `.gitignore`, as you already have these in your local clone.
* Copy the new repository's URL (e.g., `https://github.com/username/my-new-project.git`).

### 3. Update the "Remote" Origin
Currently, your local folder thinks it should push to the *old* repository. You need to point it to the *new* one.

* **Check current remote:**
  ```bash
  git remote -v
  ```
* **Change the remote URL:**
  ```bash
  git remote set-url origin https://github.com/username/my-new-project.git
  ```
* **Verify the change:**
  ```bash
  git remote -v
  ```
  *(It should now show your new repo URL).*

### 4. Optional: Clean Slate vs. Keeping History
Decide if you want to keep the commit history from the old project.

* **Option A: Keep History (Standard)**
    Just push your existing branches to the new repo:
    ```bash
    git push -u origin main
    ```

* **Option B: Fresh Start (Clean History)**
    If you want the new project to start at "Initial Commit" with no record of the old work:
    ```bash
    rm -rf .git
    git init
    git add .
    git commit -m "Initial commit for new project"
    git remote add origin https://github.com/username/my-new-project.git
    git push -u origin main
    ```

---

### 5. Final Housekeeping
Since you are starting a new project, don't forget to update the following files if they exist:
1.  **`package.json`** or **`requirements.txt`**: Update the project name and version.
2.  **`README.md`**: Update the titles and descriptions.
3.  **Environment Variables**: If you have a `.env.example`, make sure it reflects any new configurations.

---

## How to use GitHub Templates

GitHub's **Template Repository** feature is the most efficient way to manage "boilerplate" code. Instead of manually cloning and changing remotes every time, you can turn any existing repository into a blueprint for future ones.

### 1. How to Turn Your Project into a Template
1.  Navigate to your existing repository on **GitHub.com**.
2.  Click on the **Settings** tab at the top.
3.  In the **General** section, look for the checkbox labeled **Template repository**.
4.  Check that box. GitHub will automatically save the change.

### 2. How to Use the Template
Once a repo is marked as a template, a new green button appears at the top of the repository page: **"Use this template"**.

* **Clicking "Use this template"** allows you to "Create a new repository."
* **The Benefit:** It creates a brand-new repo with a **single "Initial commit"** containing all your files, but *without* the entire commit history of the original project. This keeps your new project's history clean from the start.

### 3. Advanced: Using "Repository Dispatch" for Automation
If your projects involve complex setups (like setting up an Oracle DB schema or configuring a specific Python environment), you can use **GitHub Actions** with your template.

You can include a `.github/workflows/post-create.yml` file in your template. When you create a new repo from that template, the action can automatically:
* Rename variables in your `README.md`.
* Update the `package.json` or `pyproject.toml` with the new repo name.
* Run a "Sanity Check" test suite to ensure the boilerplate is working.

### 4. Comparison: Cloning vs. Templating

| Feature | Manual Clone/Remote Change | GitHub Template |
| :--- | :--- | :--- |
| **History** | Keeps all old commits (unless deleted). | Starts with a clean "Initial commit." |
| **Speed** | Requires Terminal commands. | One click on the GitHub UI. |
| **Automation** | Manual search/replace of names. | Can be automated via GitHub Actions. |
| **Visibility** | Works for any repo you can access. | Must be explicitly marked as "Template." |

### 5. Using Template Repository in Development

Making a repository a **Template** does not change how it functions for you as a developer. You can still push code, create branches, merge Pull Requests, and run CI/CD pipelines exactly as you did before.

Think of the "Template" status as an **additional superpower** rather than a change in state.

#### How it behaves in practice:
* **Regular Development:** You continue to work in your local folder, `git push` to the origin, and manage your files.
* **The "Snapshot" Effect:** When you (or someone else) clicks **"Use this template"**, GitHub essentially takes a "snapshot" of the code exactly as it exists on your **default branch** (usually `main`) at that specific moment.
* **Future Updates:** If you add a new Python utility script or a new Oracle SQL initialization file to your template repo tomorrow, any *new* projects created from that template from that point forward will include those updates. Existing projects you already bootstrapped will **not** be affected.

#### A Tip for Your Two New Projects
Since you are bootstrapping two projects at once, here is how to handle the "Template" workflow effectively:

1.  **Finalize the Core:** Make sure your "Template" repo has the common denominator code both projects need (e.g., your MinIO connection logic or basic logging patterns).
2.  **Push to Template:** Commit those changes to GitHub.
3.  **Spawn Project A:** Click "Use this template" $\rightarrow$ Name it `Project-A`.
4.  **Spawn Project B:** Click "Use this template" $\rightarrow$ Name it `Project-B`.
5.  **Specialize:** Now, go into the local folders for A and B and add the specific agentic patterns or threat-intel modules unique to each.

#### One Small Caveat
If you use the **GitHub Action** to rename placeholders, remember that the Action runs in the *new* repositories (`Project-A` and `Project-B`), not in the template itself. This keeps your template "clean" with the placeholders intact for the next time you need it.

### Pro-Tip for Python Projects
Since you often work with many individual Python scripts grouped by patterns, you might want to include a **`setup.py`** or **`pyproject.toml`** in your template. This allows you to run `pip install -e .` immediately after creating a new project so all your internal modules are correctly pathed.

---

## How to use GitHub Actions

Since your projects often involve complex structures—like specific Python patterns, Oracle DB schemas, or threat intelligence modules—automating the "handover" from a template to a new repository saves a lot of manual renaming.

We can use **GitHub Actions** to watch for the `repository_created` event. When you create a new repo from your template, this script will automatically find placeholders and replace them with your new project's name.

### 1. The Automation Script (`.github/workflows/post-create.yml`)

Create this file in your **template repository**. It uses a simple `sed` command to swap out a placeholder (like `PROJECT_NAME_PLACEHOLDER`) with the actual name of your new repo.

```yaml
name: Post-Template Initialization
on: [push] # Fires on the first push (which happens during template creation)

jobs:
  setup-new-repo:
    if: github.run_number == 1 # Only runs once, right after creation
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Rename Placeholders
        run: |
          # Replace placeholder in README and configuration files
          find . -type f -not -path '*/.*' -exec sed -i "s/PROJECT_NAME_PLACEHOLDER/${{ github.event.repository.name }}/g" {} +
          
      - name: Commit and Push Changes
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add .
          git commit -m "chore: initialize project from template"
          git push
```

### 2. How to Set Up Your Template Files
To make the script above work, you just need to use the string `PROJECT_NAME_PLACEHOLDER` anywhere you want the new name to appear.

* **`README.md`**:
    ```markdown
    # PROJECT_NAME_PLACEHOLDER
    This project was generated from a custom pattern template.
    ```
* **`setup.py` or `pyproject.toml`**:
    ```python
    name="PROJECT_NAME_PLACEHOLDER",
    version="0.1.0",
    ```

### 3. Workflow Comparison: Template vs. Manual

| Step | Manual Method | Automated Template Method |
| :--- | :--- | :--- |
| **Creation** | `git clone` + `remote set-url` | Click "Use this template" |
| **History** | Full history remains | Clean "Initial commit" |
| **Renaming** | Manual search and replace | **Handled by GitHub Action** |
| **Dependencies** | Manual `pip install` | Action can pre-verify environment |

### Why this fits your current projects
Since you are testing different **agentic design patterns** and building a **threat intelligence network**, you likely have a specific directory structure you prefer (e.g., `/scripts`, `/data`, `/logs`). By using this template:
1.  Your **MinIO** or **ChromaDB** connection strings can stay consistent.
2.  Your **Oracle DB** initialization scripts stay in a known location.
3.  Every new experiment starts with the same clean, professional structure without you typing a single `mkdir` command.

---