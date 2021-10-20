const darkTheme = window.matchMedia('(prefers-color-scheme: dark)').matches;
const head = document.getElementsByTagName('head')[0]
const link = document.createElement('link')

link.rel = 'icon'
link.href = './favicon.ico';

head.append(link)

if (darkTheme) {
    link.href = './favicon-dark-theme.ico';
}



