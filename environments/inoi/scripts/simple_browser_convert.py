"""
Simple browser-based SVG to PNG converter.
Standalone script using only playwright.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from PIL import Image
from io import BytesIO

async def convert_svg_browser(svg_path: Path, png_path: Path) -> bool:
    """Convert SVG to PNG using headless Chrome."""
    try:
        # Read SVG
        svg_text = svg_path.read_text(encoding='utf-8', errors='ignore')

        # Build HTML
        html = f"""
        <!doctype html>
        <html>
        <head><meta charset="utf-8">
        <style>html,body{{margin:0;padding:0;background:#fff;}}</style>
        </head>
        <body>{svg_text}</body>
        </html>
        """

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(device_scale_factor=1)
            await page.set_content(html, wait_until="load")

            # Get SVG dimensions
            box = await page.evaluate("""
                () => {
                  const svg = document.querySelector('svg');
                  if (!svg) return {w: 800, h: 600};
                  const toPx = v => {
                    if(!v) return null;
                    if(String(v).endsWith('px')) return parseFloat(v);
                    const f=parseFloat(v);
                    return isNaN(f)?null:f;
                  };
                  let W = toPx(svg.getAttribute('width'));
                  let H = toPx(svg.getAttribute('height'));
                  if ((W==null || H==null) && svg.viewBox && svg.viewBox.baseVal) {
                    W=svg.viewBox.baseVal.width||800;
                    H=svg.viewBox.baseVal.height||600;
                  }
                  return {w: Math.max(1, Math.round(W||800)), h: Math.max(1, Math.round(H||600))};
                }
            """)

            W = max(1, int(round(box["w"])))
            H = max(1, int(round(box["h"])))

            # Set dimensions and crop to content
            await page.evaluate("""
                (args) => {
                  const svg = document.querySelector('svg');
                  if(!svg) return;
                  let bbox = null;
                  try {
                    if(typeof svg.getBBox === 'function') {
                      bbox = svg.getBBox();
                    }
                  } catch(e) {}
                  if(bbox && bbox.width>0 && bbox.height>0) {
                    svg.setAttribute('viewBox', bbox.x+' '+bbox.y+' '+bbox.width+' '+bbox.height);
                    svg.setAttribute('preserveAspectRatio','xMidYMid meet');
                  }
                  svg.setAttribute('width', String(args.w));
                  svg.setAttribute('height', String(args.h));
                }
            """, {"w": W, "h": H})

            await page.set_viewport_size({"width": W, "height": H})
            png_bytes = await page.locator("svg").first.screenshot(omit_background=False)
            await browser.close()

        # Save with PIL to ensure RGB format
        img = Image.open(BytesIO(png_bytes))
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            base = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(base, img.convert("RGBA")).convert("RGB")
        else:
            img = img.convert("RGB")

        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(png_path, format='PNG')

        return True

    except Exception as e:
        print(f"Error converting {svg_path.name}: {e}")
        return False


async def convert_all(svg_paths, png_paths):
    """Convert multiple SVGs with parallel execution."""
    semaphore = asyncio.Semaphore(4)  # 4 parallel workers

    async def bounded_convert(svg, png):
        async with semaphore:
            result = await convert_svg_browser(svg, png)
            return result, svg

    tasks = [bounded_convert(svg, png) for svg, png in zip(svg_paths, png_paths)]

    completed = 0
    success = 0

    for coro in asyncio.as_completed(tasks):
        result, svg = await coro
        completed += 1
        if result:
            success += 1
        print(f"Progress: {completed}/{len(tasks)} | Success: {success} | Last: {svg.name}")

    return success


async def main():
    print("="*60)
    print("Simple Browser SVG Conversion")
    print("="*60)

    assets_svg_dir = Path("assets_svg")
    assets_png_dir = Path("assets")

    # Find all SVGs in assets_svg/ that don't have PNGs in assets/
    to_convert = []
    for svg_path in assets_svg_dir.glob("*.svg"):
        png_name = svg_path.stem + ".png"
        png_path = assets_png_dir / png_name
        if not png_path.exists():
            to_convert.append((svg_path, png_path))

    print(f"\nFound {len(to_convert)} SVGs to convert")
    print("Source: assets_svg/")
    print("Target: assets/")
    print("Using 4 parallel workers...\n")

    if not to_convert:
        print("✓ All SVGs already converted!")
        return

    # Convert
    svgs, pngs = zip(*to_convert)
    success = await convert_all(svgs, pngs)

    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(to_convert)}")
    print(f"Success: {success} ({100*success/len(to_convert):.1f}%)")
    print(f"Failed: {len(to_convert) - success}")

    if success == len(to_convert):
        print("\n✓ ALL CONVERSIONS SUCCESSFUL!")
    else:
        print(f"\n⚠ {len(to_convert) - success} conversions failed")


if __name__ == "__main__":
    asyncio.run(main())
