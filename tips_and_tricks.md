[← Back to Home](/)

# Various Tips and How To...s

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
