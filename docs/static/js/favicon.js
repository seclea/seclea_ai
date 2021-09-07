const darkTheme = window.matchMedia('(prefers-color-scheme: dark)').matches;
const head = document.getElementsByTagName('head')[0]
const link = document.createElement('link')

link.rel = 'icon'
link.href = '../../static/assets/favicon.ico';

head.append(link)

if (darkTheme) {
    link.href = '../../static/assets/favicon-dark-theme.ico';
}

