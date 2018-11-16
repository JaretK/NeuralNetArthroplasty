"""
Contains attributes for different model assessment conditions
"""

from abc import ABCMeta, abstractproperty

class AbstractAttribute:
    """Abstract base class for Attributes"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def normalized_attributes(self):
        pass

    @abstractproperty
    def categorical_attributes(self):
        pass

    def get_all_attributes(self):
        l = []
        l.extend(self.normalized_attributes)
        l.extend(self.categorical_attributes)
        return l

class Attributes:
    _TCosts_2 = 'total_costs_inflation_adjusted_2'
    _LOS_2 = 'length_of_stay_2'
    _DISPOSITION_HOME_BINARY = 'patient_disposition_home'
    class _nis_attributes(AbstractAttribute):
        normalized_attributes = ['AGE', 'NCHRONIC', 'NDX',]
        categorical_attributes =[
        'apr_risk_of_mortality',
        'apr_severity_of_illness_description',
        'emergency_department_indicator',
        'ethnicity',
        'gender',
        'race',
        'type_of_admission',
        'CM_AIDS',
        'CM_ALCOHOL',
        'CM_ANEMDEF',
        'CM_ARTH',
        'CM_BLDLOSS',
        'CM_CHF',
        'CM_CHRNLUNG',
        'CM_COAG',
        'CM_DEPRESS',
        'CM_DM',
        'CM_DMCX',
        'CM_DRUG',
        'CM_HTN_C',
        'CM_HYPOTHY',
        'CM_LIVER',
        'CM_LYMPH',
        'CM_LYTES',
        'CM_METS',
        'CM_NEURO',
        'CM_OBESE',
        'CM_PARA',
        'CM_PERIVASC',
        'CM_PSYCH',
        'CM_PULMCIRC',
        'CM_RENLFAIL',
        'CM_TUMOR',
        'CM_ULCER',
        'CM_VALVE',
        'CM_WGHTLOSS',
        'AWEEKEND',
        'HOSP_LOCTEACH',
        'TRAN_IN_BINARY',
        'ZIPINC_QRTL',
        'HOSPST'
    ]

    class _custom_attributes(AbstractAttribute):
        normalized_attributes = ['AGE','NCHRONIC']
        categorical_attributes = [
            'gender',
            'race',
            'CM_CHF',
            'CM_HTN_C',
            'CM_PERIVASC',
            'CM_PSYCH',
            'CM_CHRNLUNG',
            'CM_ARTH',
            'CM_ULCER',
            'CM_LIVER',
            'CM_DM',
            'CM_NEURO',
            'CM_DMCX',
            'CM_RENLFAIL',
            'CM_TUMOR',
            'CM_METS',
            'CM_AIDS',
            'HOSPST'
        ]

if __name__ == '__main__':
    print(Attributes._nis_attributes().get_all_attributes())
