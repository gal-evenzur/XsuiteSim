import numpy as np
import matplotlib.pyplot as plt
import os
import ROOT
plt.rcParams['image.cmap'] = 'afmhot'

import pickle
pydir = os.path.dirname(os.path.abspath(__file__)) # This results ""


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kSolar)
# ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.16)
ROOT.gStyle.SetGridColor(ROOT.kGray)
ROOT.gStyle.SetGridWidth(1)

m2mm = 1000

dipoleTPL = ROOT.TPolyLine()
dipoleTPL.SetNextPoint(-22.352,-66.927)
dipoleTPL.SetNextPoint(-22.352,34.927)
dipoleTPL.SetNextPoint(22.352,34.927)
dipoleTPL.SetNextPoint(22.352,-66.927)
dipoleTPL.SetNextPoint(-22.352,-66.927)
dipoleTPL.SetLineColor(ROOT.kBlue)
dipoleTPL.SetLineWidth(1)
flangeTPL = ROOT.TPolyLine()
flangeTPL.SetNextPoint(-22.352,-63.752)
flangeTPL.SetNextPoint(-22.352,31.752)
flangeTPL.SetNextPoint(22.352,31.752)
flangeTPL.SetNextPoint(22.352,-63.752)
flangeTPL.SetNextPoint(-22.352,-63.752)
flangeTPL.SetLineColor(ROOT.kAzure+1)
flangeTPL.SetLineWidth(1)

ztitle = "Particles in full acceptance"
histD_entrance = ROOT.TH2D("histD_entrance",f"Dipole entrance plane;x_{{LAB}} [mm];y_{{LAB}} [mm];{ztitle}",200,-80,+80, 200,-70,+90)
histD_exit     = ROOT.TH2D("histD_exit",    f"Dipole exit plane;x_{{LAB}} [mm];y_{{LAB}} [mm];{ztitle}",    200,-80,+80, 200,-70,+90)


# load Data from pickle file
pickle_filename = os.path.join(pydir, 'Data.pkl')
with open(pickle_filename, 'rb') as f:
    Data = pickle.load(f)

# shape(Data) = (len(magnet_settings), n_final_alive, 4, (par))
# par = (s_values, x[i], y[i], pz[i]) [ i = particle index ]
# now par is a tuple with (s_values, x_values, y_values, pz) for each particle
# where s_values is the same for all particles, but x_values, y_values, pz are different
magset = {0:"Run 490.0", 1:"Run 490.1", 2:"Run 490.2", 3:"Run 490.5"}


def plot_histogram(x, y, bins, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(x, y, bins=bins)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_title(title)


def plot_histogram_from_data(Data, magnet_idx, monitor_idx):
    x = []
    y = []
    for par in Data[magnet_idx]:
        x.append(par[1][monitor_idx])
        y.append(par[2][monitor_idx])
    x = np.array(x)
    y = np.array(y)
    title = f"Magnet setting: {magnet_idx}, Monitor: {monitor_idx}"
    plot_histogram(x, y, bins=100, title=title)
    

def root_histogram_from_data(Data, h, magnet_idx, monitor_idx):
    for par in Data[magnet_idx]:
        x = par[1][monitor_idx]*m2mm
        y = par[2][monitor_idx]*m2mm
        print(f"x={x}, y={y}")
        h.Fill(x,y)
    return h




plot_histogram_from_data(Data, magnet_idx=3, monitor_idx=0)

plot_histogram_from_data(Data, magnet_idx=3, monitor_idx=1)

plot_histogram_from_data(Data, magnet_idx=3, monitor_idx=0)

plt.show()

root_histogram_from_data(Data, histD_exit, magnet_idx=3, monitor_idx=0)

cnv = ROOT.TCanvas("cnv","",550,500)
cnv.SetTicks(1,1)
cnv.SetGridx()
cnv.SetGridy()
histD_exit.GetZaxis().SetTitleOffset(1.3)
histD_exit.Draw("colz")
dipoleTPL.Draw()
flangeTPL.Draw()
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlack)
s.SetTextFont(22)
s.SetTextSize(0.045)
s.DrawLatex(0.17,0.88,"Xsuite MC")
#
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlack)
s.SetTextFont(132)
s.SetTextSize(0.045)
s.DrawLatex(0.17,0.83,f"#mu_{{x}}={histD_exit.GetMean(1):.1f} mm, #sigma_{{x}}={histD_exit.GetStdDev(1):.1f} mm")
s.DrawLatex(0.17,0.78,f"#mu_{{y}}={histD_exit.GetMean(2):.1f} mm, #sigma_{{y}}={histD_exit.GetStdDev(2):.1f} mm")
#
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlack)
s.SetTextFont(132)
s.SetTextSize(0.045)
s.DrawLatex(0.35,0.88,"(Global alignment: as designed)")
#
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlue)
s.SetTextFont(132)
s.SetTextSize(0.045)
s.DrawLatex(0.367,0.69,"Dipole aperture")
#
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kAzure+1)
s.SetTextFont(132)
s.SetTextSize(0.045)
s.DrawLatex(0.15,0.65,"Flange aperture")
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"Xsuite_dipole_exit.pdf")
