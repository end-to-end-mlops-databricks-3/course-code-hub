# Course Documentation - GitHub Pages Setup

This directory contains the original Word document recaps for each lecture. The GitHub Pages documentation has been created as markdown files in the root directory and `lectures/` folder.

## GitHub Pages Structure

```
├── index.md                    # Main course overview page
├── lectures/                   # Individual lecture pages
│   ├── lecture-1.md           # Lecture 1: Introduction & Code Quality
│   ├── lecture-2.md           # Lecture 2: Experiment Tracking
│   ├── lecture-3.md           # Lecture 3: Feature Engineering
│   ├── lecture-4.md           # Lecture 4: Model Deployment
│   ├── lecture-5.md           # Lecture 5: Advanced Serving
│   └── lecture-6.md           # Lecture 6: A/B Testing & Monitoring
├── _config.yml                # Jekyll configuration
├── Gemfile                    # Ruby dependencies
└── .github/workflows/pages.yml # GitHub Actions for deployment
```

## Original Recaps

The original lecture recaps are available as Word documents:
- Lecture 1 Recap.docx
- Lecture 2 Recap.docx
- Lecture 3 Recap.docx
- Lecture 4 Recap.docx
- Lecture 5 Recap.docx
- Lecture 6 Recap.docx

## Setting Up GitHub Pages

1. **Enable GitHub Pages** in your repository settings:
   - Go to Settings → Pages
   - Source: GitHub Actions
   - The workflow will automatically deploy when you push changes

2. **Local Development** (optional):
   ```bash
   # Install Ruby and Bundler
   gem install bundler jekyll
   
   # Install dependencies
   bundle install
   
   # Serve locally
   bundle exec jekyll serve
   ```

3. **Access Your Site**:
   - URL: `https://end-to-end-mlops-databricks-3.github.io/course-code-hub3`
   - Updates automatically when you push to main branch

## Customization

- **Theme**: Modify `theme` in `_config.yml`
- **Navigation**: Update `header_pages` in `_config.yml`
- **Content**: Edit markdown files in `lectures/` directory
- **Styling**: Add custom CSS in `assets/css/style.scss`

## Content Migration

To migrate content from Word documents to markdown:
1. Copy text from Word documents
2. Convert to markdown format
3. Add to appropriate lecture files
4. Update links and references
5. Test locally before pushing
