# Publishing checklist

- Both companion modules are included. Confirm their provenance and any license obligations.
- Confirm the rights to redistribute every source file and companion module.
- MIT license text is included as `LICENSE`, with copyright (c) 2026 Kwangjo Kim. Preserve the copyright and permission notice in redistributed copies.
- Add the confirmed paper title, full author list, affiliation information as desired, and the publication or preprint URL. Add `CITATION.cff` after these details are finalized; avoid invented DOI or authorship metadata.
- Run all three default experiments with the companions available and retain the complete logs. Publish selected logs if desired; `results/` is ignored by default, so intentionally add any chosen public logs.
- Check the README's author-reported counts against the retained logs.
- Create a GitHub repository, upload this folder's contents, and verify that README tables and code blocks render correctly. Upload the contents rather than the enclosing ZIP if you want GitHub to show the README on the repository homepage.
- Record the release commit/tag for the manuscript's reproducibility statement.

The original script bytes are preserved in this package. Filenames have only had the upload suffix `(1)` removed. Header comments in the supplied 128-bit and 192-bit files describe earlier names; they may be cleaned up separately before release.
