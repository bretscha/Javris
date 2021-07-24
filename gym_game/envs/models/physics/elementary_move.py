import math
from .physics import * 
class ElementaryMove:
    def __init__(self, fRpj, distance, t,speedFactor = 1.0) :
        self.m_fRpj = fRpj
        self.m_overall_distance = distance
        self.m_t = t
        self.m_speedFactor = speedFactor
        self.m_current_phase = 0.0
        self.m_shift = 0.0
        self.m_timeShift = 0.0
        self.m_distanceShift = 0.0
        self.m_deacceleration_phase = 0.5
        self.m_duration

        x = abs(self.m_overall_distance)
        # http://www.colby.edu/chemistry/PChem/scripts/lsfitpl.html
        self.m_duration = 2.994 * x - 0.000355 / x + 0.3288
        self.m_duration = 0.1 if self.m_duration < 0.1 else self.m_duration

        self.m_duration /= speedFactor        

    def getDelta(self, t):
        assert t >= self.m_t

        self.m_current_phase = self.get_phase(t - self.m_t + self.m_timeShift)     
        shift = self.m_overall_distance * GetQuad(self.m_current_phase) - self.m_distanceShift
        delta = shift - self.m_shift
        self.m_shift = shift

        # Deaccelerate 
        if ((self.m_current_phase > self.m_deacceleration_phase) and  (self.m_timeShift == 0) and (self.m_current_phase <= 0.5)):
            self.m_timeShift = self.m_duration - 2 * (t - self.m_t)
            phase1 = self.get_phase(t - self.m_t + self.m_timeShift)
            phase2 = self.get_phase(t - self.m_t)
            self.m_distanceShift = self.m_overall_distance * (GetQuad(phase1) - GetQuad(phase2))

        return delta

    # Returns True if distance==0  or the time passed(m_phase) == 100%
    def is_finished(self):
        if (abs(self.m_overall_distance) == 0): return True
        return (not(self.m_current_phase <= 1))

        
    def get_Min_Deacc_Distance(self, t):
        return self.m_overall_distance * (1 - self.m_current_phase) if self.m_current_phase > 0.5 else self.m_overall_distance * self.m_current_phase

    # Returns the remaining distance to the last point in % 
    def GetRemainingDistance(self):
        #  if no deceleration initiated
        if (self.m_deacceleration_phase >= 0.5):
            return self.m_overall_distance - self.m_overall_distance * GetQuad(self.m_current_phase)
        # deceleration initiated but not started
        else:
            if (self.m_current_phase < self.m_deacceleration_phase):
                return 2 * self.m_overall_distance * GetQuad(self.m_deacceleration_phase) - self.m_overall_distance * GetQuad(self.m_current_phase);
            else:
                return self.m_overall_distance - self.m_overall_distance * GetQuad(self.m_current_phase)

    # returns the time passed in % time_since_start / m_duration
    def get_phase(self, time_since_start):
        phase = 0.0 if abs(self.m_overall_distance) == 0 else time_since_start / self.m_duration
        return 1.0 if (phase > 1.0) else phase

    #  Try to achieve complete stop in distance d
    def SetupDeacc(self, deacceleration_distance):
		# The decelaration has been already started and cannot be intensified
        if (self.m_current_phase > 0.5): return
		#  We cannot deaccelarate faster than we have accelarated
        current_dist = self.m_overall_distance * GetQuad(self.m_current_phase)
        deacceleration_distance =  current_dist  if (abs(deacceleration_distance) < abs(current_dist)) else deacceleration_distance
		# phase when the deaccelaration should be started
        progress = abs((current_dist + deacceleration_distance) / self.m_overall_distance / 2.0)
        progress = 1 if progress > 1 else progress
        self.m_deacceleration_phase = GetQuadInv(progress)
        self.m_deacceleration_phase = 0.5 if self.m_deacceleration_phase > 0.5 else self.m_deacceleration_phase

    def StopAsap(self):
        self.SetupDeacc(0)