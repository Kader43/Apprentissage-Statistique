\LoadClass[a4paper,12pt]{report} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage[french]{babel} % Language of the document
\RequirePackage[utf8]{inputenc}
\RequirePackage[section]{placeins}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathtools}
\RequirePackage{siunitx}
\RequirePackage{float}
\RequirePackage{graphicx}
\RequirePackage[justification=centering]{caption}
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
\RequirePackage{fancyhdr}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry}

%-------------------- Report information ----------------------

\newcommand{\subject}[1]{\renewcommand{\subject}{#1}}
\newcommand{\topic}[1]{\renewcommand{\topic}{#1}}
\newcommand{\ttitle}[1]{\renewcommand{\ttitle}{#1}}
\newcommand{\professor}[1]{\renewcommand{\professor}{#1}}
\newcommand{\students}[1]{\renewcommand{\students}{#1}}
\newcommand{\logo}[1]{\renewcommand{\logo}{#1}}
\newcommand{\uni}[1]{\renewcommand{\uni}{#1}}

\newcommand{\buildmargins}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.25]{logos/ssd_logo.png}}
\rhead{\nouppercase{\leftmark}}

\lfoot{\subject}
}

\newcommand{\buildcover}{
\begin{titlepage}

	\centering
	\includegraphics[width=0.3\textwidth]{\logo}\par\vspace{1cm}
	
	{\scshape\LARGE \uni \par}
	\vspace{1cm}
	
	\rule{\linewidth}{0.3mm} \\[0.4 cm]
	{\Huge\bfseries \ttitle \par}
	\rule{\linewidth}{0.3mm} \\[1.5 cm]

    {\LARGE \textbf{Vecteur Support Machine} \par}
    \vspace{1.5cm}
	
	\begin{minipage}{0.5\textwidth}
		\begin{flushleft} \large
		\emph{\textbf{Etudiant :}}\\
        \students\\
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Encadrant :}} \\
		 \professor \\
		\end{flushright}
	\end{minipage}
    
	\vfill
	{\large 2024--2025\par}

\end{titlepage}
}


\newcommand{\toc}{
\tableofcontents
\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}



