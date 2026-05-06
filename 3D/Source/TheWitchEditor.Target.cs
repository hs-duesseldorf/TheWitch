using UnrealBuildTool;
using System.Collections.Generic;

public class TheWitchEditorTarget : TargetRules
{
	public TheWitchEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V6;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("TheWitch");
	}
}