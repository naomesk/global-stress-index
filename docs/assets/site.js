(function(){
  // Repo link: https://USER.github.io/REPO/ -> https://github.com/USER/REPO
  var h=location.hostname.match(/^([^.]+)\.github\.io$/), p=location.pathname.split('/')[1];
  var url=(h&&p)?'https://github.com/'+h[1]+'/'+p:null;
  document.querySelectorAll('[data-repo]').forEach(function(a){
    if(url){a.href=url;} else {a.removeAttribute('href');a.title='Shown once the site is on GitHub Pages';a.style.opacity='.6';}
  });
  document.querySelectorAll('[data-repo-url]').forEach(function(el){
    el.textContent=url||'https://github.com/YOUR-USER/global-stress-index';
  });
  // Copy buttons on code blocks
  document.querySelectorAll('pre').forEach(function(pre){
    var b=document.createElement('button');b.className='copy';b.type='button';b.textContent='Copy';
    b.addEventListener('click',function(){
      var t=pre.querySelector('code').innerText;
      (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).then(function(){b.textContent='Copied';setTimeout(function(){b.textContent='Copy'},1500)},function(){b.textContent='Select and copy'});
    });
    pre.appendChild(b);
  });
})();
