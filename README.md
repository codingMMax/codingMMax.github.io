# Academic Website Template

A clean, modern academic website template based on Sixun Dong's homepage design.

## 🎯 Features

- **Minimal & Clean Design**: Focus on content with modern typography
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Academic-Focused**: Optimized for researchers and academics
- **Blog Support**: Integrated blog with cover images, tags, and categories
- **Fast Loading**: Pure HTML/CSS/JS, no heavy frameworks

## 🏗️ Structure

```
.
├── index.html          # Main homepage (Bio)
├── publications.html   # Publications page  
├── blog.html          # Blog listing page
├── styles.css         # Main stylesheet
├── blog.css           # Blog-specific styles
├── images/            # Profile and blog images
│   ├── profile.jpg    # Your profile photo
│   └── blog/          # Blog cover images
├── files/             # CV and other files
├── posts/             # Individual blog posts
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Local Preview

Simply open `index.html` in your browser to preview the site locally.

**Or use a simple HTTP server:**

```bash
# Python 3
python -m http.server 8000

# Python 2  
python -m SimpleHTTPServer 8000

# Node.js (if you have http-server installed)
npx http-server

# Then open http://localhost:8000
```

### 2. Customize Content

#### Profile Information
Edit `index.html`:
- Replace profile photo: `images/profile.jpg`
- Update name, title, affiliation
- Modify the bio and research interests
- Update contact links

#### Publications
Edit `publications.html`:
- Add your papers in the appropriate sections
- Update publication venues and links
- Modify the statistics section

#### Blog
- Add blog posts in the `blog.html` grid
- Create individual post files in `posts/` directory
- Add cover images to `images/blog/`

### 3. Deploy to GitHub Pages

1. **Upload to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/yourusername.github.io.git
git push -u origin main
```

2. **Enable GitHub Pages:**
   - Go to repository Settings
   - Navigate to Pages section  
   - Select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Your site will be available at `https://yourusername.github.io`

## 📝 Content Guidelines

### Adding Publications

Each publication should include:
- Venue badge with appropriate styling
- Full title and author list (bold your name)
- Brief summary/abstract
- Links to paper, code, video, etc.

### Writing Blog Posts

For each blog post:
1. Add an entry in `blog.html`
2. Create the actual post file in `posts/`
3. Add a cover image to `images/blog/`
4. Include appropriate tags and metadata

Example blog post structure:
```html
<article class="blog-card">
    <div class="blog-image">
        <img src="images/blog/post-cover.jpg" alt="Post Title">
        <div class="blog-category">Category</div>
    </div>
    <div class="blog-content">
        <time class="blog-date">Date</time>
        <h2 class="blog-title">
            <a href="posts/post-name.html">Post Title</a>
        </h2>
        <p class="blog-excerpt">Brief description...</p>
        <div class="blog-tags">
            <span class="tag">Tag1</span>
            <span class="tag">Tag2</span>
        </div>
        <div class="blog-meta">
            <span class="read-time">X min read</span>
            <a href="posts/post-name.html" class="read-more">Read More →</a>
        </div>
    </div>
</article>
```

## 🎨 Customization

### Colors
The color scheme is defined in CSS variables in `styles.css`:
```css
:root {
    --color-primary: #1a1a1a;      /* Main text */
    --color-secondary: #666666;     /* Secondary text */
    --color-accent: #2563eb;        /* Links and highlights */
    --color-background: #ffffff;    /* Page background */
    --color-surface: #f8fafc;       /* Card backgrounds */
    --color-border: #e2e8f0;        /* Borders */
}
```

### Typography
The template uses Inter font. You can change it by updating the Google Fonts link and the CSS variable:
```css
--font-family: 'Your Font', sans-serif;
```

### Layout
The main layout uses CSS Grid for the hero section:
- Left column: Profile photo (300px fixed width)
- Right column: Bio and information (flexible)

On mobile, it switches to a single column layout.

## 📱 Mobile Responsiveness

The template includes responsive breakpoints:
- **Desktop**: > 1024px (full layout)
- **Tablet**: 768px - 1024px (adjusted spacing)
- **Mobile**: < 768px (stacked layout)
- **Small Mobile**: < 480px (compact design)

## 🔧 Advanced Features

### SEO Optimization
- Semantic HTML structure
- Proper meta tags
- Descriptive alt texts
- Structured data ready

### Performance
- Minimal CSS and JavaScript
- Optimized images (you should compress them)
- No external dependencies except Google Fonts

### Accessibility
- Proper heading hierarchy
- Focus styles for keyboard navigation
- Color contrast compliance
- Screen reader friendly

## 📋 Required Images

Make sure to add these images:
- `images/profile.jpg` - Your profile photo (240x240px recommended)
- `images/blog/*.jpg` - Blog cover images (400x200px recommended)

## 🤝 Contributing

Feel free to fork this template and make it your own! If you make improvements, consider sharing them back.

## 📄 License

This template is free to use for academic purposes. Attribution appreciated but not required.

---

**Happy coding!** 🚀

For questions or issues, feel free to open an issue on GitHub.
