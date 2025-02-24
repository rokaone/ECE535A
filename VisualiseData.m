%Dependencies:

%export_fig from https://uk.mathworks.com/matlabcentral/fileexchange/23629-export_fig
%This package is absolutely essential as it allows high res export of MATLAB figures

%First, load up the data we've generated

clear
load('MonteCarloReportData.mat')

set(0,'DefaultAxesFontSize',8); %Eight point Times is suitable typeface for an IEEE paper. Same as figure caption size
set(0,'DefaultFigureColor','w')
set(0,'defaulttextinterpreter','tex') %Allows us to use LaTeX maths notation
set(0, 'DefaultAxesFontName', 'times');
    
figure  %Let's make a simple time series plot of notional data
set(gcf, 'Units','centimeters')

%Set figure total dimension
%set(gcf, 'Position',[0 0 8.89 4]) %Absolute print dimensions of figure. 8.89cm is essential here as it is the linewidth of a column in IEEE format
%Height can be adusted as suits, but try and be consistent amongst figures for neatness
%[pos_from_left, pos_from_bottom, fig_width, fig_height]

hold on
for n = 1:size(p_error_rep, 1)
    disp(n)
    loglog(sigmas,p_error_rep(n,:), 'LineWidth', 2, 'DisplayName', ['N = ',num2str(n*2 - 1)]); %Plot as paired data, so we're explicity stipulating the time index
end
set(gca, 'XScale', 'log', 'YScale', 'log'); % Force log scale back
hold off
legend()

%axis([-3 3 0 1]) %Note symetric vertical axis +- 2kV around the nominal 20kV level

set(gca,'YTick',[10e-5,10e-4,10e-3,10e-2,10e-1]) %Now impose sensible tickmark locations
%Let's pretend that the undervoltage limit is 19.2 kV, and the overvoltage is 21.25 kV

% set(gca,'YTickLAbel',{'18 kV', 'V_{min}', '20 kV', 'V_{max}', '22 kV'}) %Now put in informative labels at these tickmarks

%Now sort out the horizontal axes: it needs to be shown in wallclock units

set(gca,'XTick',[1e-1,2e-1,3e-1,4e-1,5e-1]) %Now impose sensible tickmark locations

% set(gca,'XTickLAbel',{'00:00', '08:00', '16:00','00:00', '08:00', '16:00', '00:00'}) %consistent tick interval


%Set size and position of axes plotting area within figure dimensions
%It is nice to keep the vertical axes aligned for multiple figures, so be consistent with the horizontal positioning of axes 
%set(gca, 'Units','centimeters')
%set(gca, 'Position',[2 0.9 6.5 2.9]) %This is the relative positioning of the axes within the frame. 
%[inset_from_left, inset_from_bottom, axes_width, axes_height]

set(gca, 'XGrid', 'on')
box off %Removes the borders of the plot area
% 
ylabel('P_e',  'FontWeight', 'bold') %Note cell matrices for line breaks
% 
set(get(gca,'YLabel'),'Rotation',0, 'VerticalAlignment','middle', 'HorizontalAlignment','right') %Tidy it with right orientation (If all our vertical axes have the same internal offset all our axis labels will be neatly aligned

set(gca,'fontsize', 14) 

xlabel('Noise power (\sigma^2)',  'FontWeight', 'bold') %Note use of unicode arrow for clarity

%Now ready for export