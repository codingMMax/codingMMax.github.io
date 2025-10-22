// Auto-generated blog data
// This file is automatically updated by build scripts
// Do not edit manually

window.BLOG_DATA = [];

// Helper function to get blog post by ID
window.getBlogPost = function(id) {
  return window.BLOG_DATA.find(post => post.id === id);
};

// Helper function to get all blog posts
window.getAllBlogPosts = function() {
  return window.BLOG_DATA;
};

console.log('Blog data loaded: ' + window.BLOG_DATA.length + ' posts');
